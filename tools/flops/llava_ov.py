import math

def compute_flops(D: int, S: int, L: int, H: int, G: int, D_ff: int) -> int:
    """
    Compute the FLOPs for a transformer model.

    Args:
        D (int): Dimension of the model.
        S (int): Sequence length.
        L (int): Number of layers.
        H (int): Number of heads.
        D_ff (int): Dimension of the feed-forward network.
        G (int): Number of Key Value Groups.

    Returns:
        int: Total FLOPs in the model.
    """
    flops = L * ((2 * S * D**2 + 2 * S * D**2 * (G / H) + 2 * S**2 * D) + 3 * S * D * D_ff)
    return flops

# NOTE: We use 64 frames in efficiency evaluation rather than 32.
# The following FLOPs are computed for 32 frames.
S = 32 * 196
L = 28
D = 3584
H = 28
G = 4
D_ff = 18944
vanilla_flops = compute_flops(D, S, L, H, G, D_ff)
print(f"LLaVA-OneVision theoretical computational cost: {vanilla_flops / 1e12:.2f} TFLOPs")

# flashvid flops.
K = 20
R = 0.10 # Retention ratio.
EXPANSION = 1.25 # Expansion ratio.
LLM_R = 0.3 # LLM retention ratio.
S1 = 32 * math.ceil(196 * R * EXPANSION)
S2 = math.ceil(S1 * LLM_R)
flashvid_flops = compute_flops(D, S1, K, H, G, D_ff) + compute_flops(D, S2, L - K, H, G, D_ff)
print(f"FlashVID theoretical computational cost: {flashvid_flops / 1e12:.2f} TFLOPs")
