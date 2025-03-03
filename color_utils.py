import matplotlib.pyplot as plt

# translate hex into rgba:
def hex_to_rgba(hex_color, alpha=1.0):
    """Converts a hex color code to an RGBA tuple.

    Args:
        hex_color: The hex color code (e.g., "#RRGGBB" or "RRGGBB").
        alpha: The alpha value (default is 1.0 for opaque).

    Returns:
        An RGBA tuple (r, g, b, a) with values ranging from 0 to 1.
    """
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    return str(r) + ' ' + str(g) + ' ' + str(b) + ' ' + str(alpha)

def rgba_color_list(length):
    """Returns a list of colors from the default matplotlib color cycle.

    Args:
        length: The number of colors to return.

    Returns:
        A list of string RGBA colors from the default matplotlib color cycle.
    """
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = [hex_to_rgba(color) for color in colors]
    return colors[:length]

