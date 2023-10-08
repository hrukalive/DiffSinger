import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import MultipleLocator

DIMENSIONS = {
    'spec': lambda x: (12, 9),
    'dur': lambda x: (max(12, min(48, x // 2)), 8),
    'curve': lambda x: (max(12, min(24, x // 256)), 8),
}

def get_bitmap_size(figure_kind, val=None):
    assert figure_kind in DIMENSIONS
    fig = plt.figure(figsize=DIMENSIONS[figure_kind](val))
    return figure_to_image(fig).shape


def spec_to_figure(spec, vmin=None, vmax=None, title=''):
    if isinstance(spec, torch.Tensor):
        spec = spec.cpu().numpy()
    fig = plt.figure(figsize=DIMENSIONS['spec'](...))
    plt.pcolor(spec.T, vmin=vmin, vmax=vmax)
    plt.title(title, fontsize=22)
    plt.tight_layout()
    return fig


def dur_to_figure(dur_gt, dur_pred, txt, title=''):
    if isinstance(dur_gt, torch.Tensor):
        dur_gt = dur_gt.cpu().numpy()
    if isinstance(dur_pred, torch.Tensor):
        dur_pred = dur_pred.cpu().numpy()
    dur_gt = dur_gt.astype(np.int64)
    dur_pred = dur_pred.astype(np.int64)
    dur_gt = np.cumsum(dur_gt)
    dur_pred = np.cumsum(dur_pred)
    fig = plt.figure(figsize=DIMENSIONS['dur'](len(txt)))
    plt.vlines(dur_pred, 12, 22, colors='r', label='pred')
    plt.vlines(dur_gt, 0, 10, colors='b', label='gt')
    for i in range(len(txt)):
        shift = (i % 8) + 1
        plt.text((dur_pred[i-1] + dur_pred[i]) / 2 if i > 0 else dur_pred[i] / 2, 12 + shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.text((dur_gt[i-1] + dur_gt[i]) / 2 if i > 0 else dur_gt[i] / 2, shift, txt[i],
                 size=16, horizontalalignment='center')
        plt.plot([dur_pred[i], dur_gt[i]], [12, 10], color='black', linewidth=2, linestyle=':')
    plt.yticks([])
    plt.xlim(0, max(dur_pred[-1], dur_gt[-1]))
    plt.title(title, fontsize=22)
    plt.legend()
    plt.tight_layout()
    return fig


def pitch_note_to_figure(pitch_gt, pitch_pred=None, note_midi=None, note_dur=None, note_rest=None):
    if isinstance(pitch_gt, torch.Tensor):
        pitch_gt = pitch_gt.cpu().numpy()
    if isinstance(pitch_pred, torch.Tensor):
        pitch_pred = pitch_pred.cpu().numpy()
    if isinstance(note_midi, torch.Tensor):
        note_midi = note_midi.cpu().numpy()
    if isinstance(note_dur, torch.Tensor):
        note_dur = note_dur.cpu().numpy()
    if isinstance(note_rest, torch.Tensor):
        note_rest = note_rest.cpu().numpy()
    fig = plt.figure()
    if note_midi is not None and note_dur is not None:
        note_dur_acc = np.cumsum(note_dur)
        if note_rest is None:
            note_rest = np.zeros_like(note_midi, dtype=np.bool_)
        for i in range(len(note_midi)):
            # if note_rest[i]:
            #     continue
            plt.gca().add_patch(
                plt.Rectangle(
                    xy=(note_dur_acc[i-1] if i > 0 else 0, note_midi[i] - 0.5),
                    width=note_dur[i], height=1,
                    edgecolor='grey', fill=False,
                    linewidth=1.5, linestyle='--' if note_rest[i] else '-'
                )
            )
    plt.plot(pitch_gt, color='b', label='gt')
    if pitch_pred is not None:
        plt.plot(pitch_pred, color='r', label='pred')
    plt.gca().yaxis.set_major_locator(MultipleLocator(1))
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    return fig


def curve_to_figure(curve_gt, curve_pred=None, curve_base=None, grid=None, title=''):
    if isinstance(curve_gt, torch.Tensor):
        curve_gt = curve_gt.cpu().numpy()
    if isinstance(curve_pred, torch.Tensor):
        curve_pred = curve_pred.cpu().numpy()
    if isinstance(curve_base, torch.Tensor):
        curve_base = curve_base.cpu().numpy()
    fig = plt.figure(figsize=DIMENSIONS['curve'](curve_gt.shape[0]))
    if curve_base is not None:
        plt.plot(curve_base, color='g', label='base')
    plt.plot(curve_gt, color='b', label='gt')
    if curve_pred is not None:
        plt.plot(curve_pred, color='r', label='pred')
    if grid is not None:
        plt.gca().yaxis.set_major_locator(MultipleLocator(grid))
    plt.grid(axis='y')
    plt.legend()
    plt.title(title, fontsize=22)
    plt.tight_layout()
    return fig


def distribution_to_figure(title, x_label, y_label, items: list, values: list, zoom=0.8):
    fig = plt.figure(figsize=(int(len(items) * zoom), 10))
    plt.bar(x=items, height=values)
    plt.tick_params(labelsize=15)
    plt.xlim(-1, len(items))
    for a, b in zip(items, values):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=15)
    plt.grid()
    plt.title(title, fontsize=30)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    return fig

def figure_to_image(figures, close=True):
    """Render matplotlib figure to numpy format.

    Note that this requires the ``matplotlib`` package.

    Args:
        figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
        close (bool): Flag to automatically close the figure

    Returns:
        numpy.array: image in [CHW] order
    """
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        import matplotlib.backends.backend_agg as plt_backend_agg
    except ModuleNotFoundError:
        print('please install matplotlib')

    def render_to_rgb(figure):
        canvas = plt_backend_agg.FigureCanvasAgg(figure)
        canvas.draw()
        data = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        w, h = figure.canvas.get_width_height()
        image_hwc = data.reshape([h, w, 4])[:, :, 0:3]
        image_chw = np.moveaxis(image_hwc, source=2, destination=0)
        if close:
            plt.close(figure)
        return image_chw

    if isinstance(figures, list):
        images = [render_to_rgb(figure) for figure in figures]
        return np.stack(images)
    else:
        image = render_to_rgb(figures)
        return image
