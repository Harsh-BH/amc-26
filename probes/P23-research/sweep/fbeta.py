def F(tp, fp, n):
    tp = tp.astype(float); fp = fp.astype(float); n = n.astype(float)
    out = np.zeros_like(tp)
    z = n == 0
    out[z] = (tp[z] + fp[z] == 0).astype(float)
    m = (~z) & (tp > 0)
    P = tp[m] / (tp[m] + fp[m]); R = tp[m] / n[m]
    out[m] = 1.25 * P * R / (0.25 * P + R)
    return out
