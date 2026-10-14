import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineActivation(nn.Module):
    def __init__(self, num_channels, num_features, grid_min = -1.0,
                        grid_max = 1.0, num_intervals = 5, degree = 3):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.num_intervals = num_intervals
        self.degree = degree
        self.num_bases = self.num_intervals + self.degree

        # Coefficients for each basis function
        self.coefficients = nn.Parameter(torch.randn(self.num_channels,
                        self.num_features, self.num_bases))

        # Create a uniform grid of knots and pad both ends to account
        # for higher order splines
        res = (grid_max - grid_min) / num_intervals
        grid = torch.linspace((grid_min - (self.degree * res)),
                        (grid_max + (self.degree * res)),
                        steps = (self.num_intervals + (2 * self.degree) + 1))
        self.register_buffer("grid", grid)


    def forward(self, x):
        # Compute basis functions
        bases = self._compute_bases(x)
       
        # Linearly combine bases using coefficients
        y = torch.sum((bases * self.coefficients.unsqueeze(0)), dim = -1)
        return y

    def _compute_bases(self, x):
        x = x.unsqueeze(-1)
 
        # Initialize order 0 bases
        knots_left = self.grid[: -1]
        knots_right = self.grid[1 :]
        bases = ((x >= knots_left) & (x < knots_right)).float()
       
        # Cox-de Boor recursion
        for ii in range(1, (self.degree + 1)):
            # Fetch relevant knots for the current recursion level
            knots_left = self.grid[: -ii - 1]
            knots_mid = self.grid[ii : -1]
            knots_right = self.grid[ii + 1 :]
           
            # Left-hand recursion term
            denom_left = knots_mid - knots_left
            term_left = ((x - knots_left) / denom_left) * bases[..., : -1]
           
            # Right-hand recursion term
            denom_right = knots_right - knots_mid
            term_right = ((knots_right - x) / denom_right) * bases[..., 1 :]
            bases = term_left + term_right
           
        # Return correct number of bases
        return bases[..., : self.num_bases]


class SepConv(nn.Module):
    def __init__(self, num_channels, kernel_size = 3, separation_number = 3):
        super().__init__()
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.separation_number = separation_number

        # Weights for the convolution
        self.weights = nn.Parameter(torch.randn(1, 1, self.kernel_size))

        # Create a non-parameter probability vector
        probs = torch.full([self.num_channels], (1.0 / self.separation_number))
        self.register_buffer('probs', probs)

    def forward(self, x):
        # Bernoulli-sample the input channels
        samples = (torch.bernoulli(self.probs) > 0.5)

        # Handle the degenerate case of no successful trials (no 1's).
        # In this case, just return the input.
        if (not samples.any()):
            return x

        x_selected = x[:, samples, :]
        x_nonselected = x[:, ~samples, :]

        # Repeat the weights to match the number of selected channels
        num_chan = x_selected.size(1)
        w = self.weights.repeat(num_chan, num_chan, 1) / num_chan

        # Compute the convolution (filter output)
        y_selected = F.conv1d(x_selected, w, padding = 'same')

        # Concatenate the result with the non-selected input channels
        y_out = torch.cat((y_selected, x_nonselected), dim = 1)
        return y_out


class APAConv(nn.Module):
    def __init__(self, num_channels, num_features,
                        spline_intervals = 5, spline_degree = 3):
        super().__init__()
        self.num_channels = num_channels
        self.num_features = num_features
        self.spline_intervals = spline_intervals
        self.spline_degree = spline_degree

        # Weights for the convolution with kernel size of 1
        self.weights = nn.Parameter(torch.randn(self.num_channels,
                        self.num_channels, 1))

        # B-Spline activation
        self.spline = BSplineActivation(self.num_channels, self.num_features,
                        num_intervals = self.spline_intervals,
                        degree = self.spline_degree)

        # SiLU activation
        self.silu = nn.SiLU()

        # Batch Norm
        self.batch_norm = nn.BatchNorm1d(self.num_channels)

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Multiply input by weights
        y_kernel = x.unsqueeze(1) * self.weights.unsqueeze(0)

        # Apply pre-activation functions and sum the results
        y_act = self.spline(y_kernel) + self.silu(y_kernel)

        # Sum across input channels
        # Note that the BSRNet paper has a double sum here, one over
        # input channels and one over features.  This makes no sense to
        # me, so I assume the inner sum over features is supposed to
        # be a correlation between the features and the kernel.
        # Since the kernel is size 1, there is no sum over features.
        y_sum = y_act.sum(dim = 2)

        # Apply batch norm and ReLU activation
        y_out = self.relu(self.batch_norm(y_sum))
        return y_out


class MLP(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels

        self.mlp = nn.Sequential(
            nn.Conv1d(self.num_channels, self.num_channels, kernel_size = 1),
            nn.BatchNorm1d(self.num_channels),
            nn.ReLU(),
            nn.Conv1d(self.num_channels, self.num_channels, kernel_size = 1)
        )

    def forward(self, x):
        y = self.mlp(x)
        return y


class BSRStage(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, num_features,
                        separation_number = 3, spline_intervals = 5,
                        spline_degree = 3):
        super().__init__()
        self.in_channels = in_channels
        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.separation_number = separation_number
        self.spline_intervals = spline_intervals
        self.spline_degree = spline_degree

        self.in_conv = nn.Conv1d(self.in_channels, self.mid_channels,
                        kernel_size = 1)
        self.sep_conv = nn.Sequential(
            SepConv(self.mid_channels,
                        separation_number = self.separation_number),
            MLP(self.mid_channels)
        )
        self.apa_conv = APAConv(self.mid_channels, self.num_features,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree)
        self.out_conv = nn.Conv1d((2 * self.mid_channels), self.out_channels,
                        kernel_size = 1)

    def forward(self, x):
        y_in = self.in_conv(x)
        y_mid1 = self.sep_conv(y_in)
        y_mid2 = self.apa_conv(y_in)
        y_cat = torch.cat((y_mid1, y_mid2), dim = 1)
        y_out = self.out_conv(y_cat)
        return y_out


class BSRNet(nn.Module):
    def __init__(self, num_features, num_categories, separation_number = 3,
                        spline_intervals = 5, spline_degree = 3):
        super().__init__()
        self.num_features = num_features
        self.num_categories = num_categories
        self.separation_number = separation_number
        self.spline_intervals = spline_intervals
        self.spline_degree = spline_degree

        # This is arbitrary, but should be a small odd number > 1
        self.max_pool_kernel = 5

        # Stage 1 (x 2)
        self.stage1 = nn.Sequential(
            BSRStage(1, 16, 64, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            BSRStage(64, 16, 64, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            nn.MaxPool1d(kernel_size = self.max_pool_kernel, stride = 1,
                        padding = (self.max_pool_kernel // 2))
        )

        # Stage 2 (x 3)
        self.stage2 = nn.Sequential(
            BSRStage(64, 32, 128, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            BSRStage(128, 32, 128, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            BSRStage(128, 32, 128, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            nn.MaxPool1d(kernel_size = self.max_pool_kernel, stride = 1,
                        padding = (self.max_pool_kernel // 2))
        )

        # Stage 3 (x 4)
        self.stage3 = nn.Sequential(
            BSRStage(128, 64, 256, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            BSRStage(256, 64, 256, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            BSRStage(256, 64, 256, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            BSRStage(256, 64, 256, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            nn.MaxPool1d(kernel_size = self.max_pool_kernel, stride = 1,
                        padding = (self.max_pool_kernel // 2))
        )

        # Stage 4 (x 2)
        self.stage4 = nn.Sequential(
            BSRStage(256, 128, 512, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            BSRStage(512, 128, 512, num_features = self.num_features,
                        separation_number = self.separation_number,
                        spline_intervals = self.spline_intervals,
                        spline_degree = self.spline_degree),
            nn.MaxPool1d(kernel_size = self.max_pool_kernel, stride = 1,
                        padding = (self.max_pool_kernel // 2))
        )

        # Basically compute the mean over features
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Final fully connected linear layers
        self.linear = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU6(),
            nn.Linear(256, self.num_categories)
        )

    def forward(self, x):
        y1 = self.stage1(x.unsqueeze(1))
        y2 = self.stage2(y1)
        y3 = self.stage3(y2)
        y4 = self.stage4(y3)
        y_flat = self.avg_pool(y4).squeeze(-1)
        y_out = self.linear(y_flat)
        return y_out


# Just a sanity check
if (__name__ == '__main__'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    bsr = BSRNet(128, 12).to(device)
    trainable = sum(param.numel() for param in bsr.parameters() if param.requires_grad)
    print(f'Trainable Parameters:  {trainable}')
    x = torch.randn(8, bsr.num_features, device = device)
    with torch.no_grad():
        y = bsr(x).cpu().numpy()
    print(f'Sanity Check:  {y.shape} ?= ({x.size(0)}, {bsr.num_categories})')

