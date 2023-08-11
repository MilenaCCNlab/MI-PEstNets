from scipy.stats import spearmanr
import plotnine as gg


def plot_recovery(param_all, label):
    """Plot a regression line for parameter recovery results.

    Args:
        param_all: A dict contains latent parameter names to its predicted and true values.
        label: The parameter name to be plotted

    Returns:
        A regression plot of parameter recovery
    """
    true_l, dl_l = f"true_{label}", f"dl_{label}"

    r_value, p_value = spearmanr(param_all[true_l], param_all[dl_l])
    r_value = round(r_value, 2)
    p_value = round(p_value, 2)
    annotated_xp = param_all[true_l].mean()
    annotated_yp = param_all[dl_l].min()

    return (
        gg.ggplot(param_all, gg.aes(x=true_l, y=dl_l))
        + gg.geom_point(color="blue")
        + gg.stat_smooth(method="lm")
        + gg.geom_line(param_all, gg.aes(x=true_l, y=true_l), color="red", size=1.2)
        + gg.annotate(
            "label",
            x=annotated_xp,
            y=annotated_yp,
            label=f"R={r_value}, p={p_value}",
            size=9,
            color="#252525",
            label_size=0,
            fontstyle="italic",
        )
    )
