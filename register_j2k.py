from numcodecs.abc import Codec
from numcodecs.compat import ensure_ndarray, ensure_bytes
from numcodecs.compat import ndarray_copy
from numcodecs.registry import register_codec
import glymur
import tempfile

class j2k(Codec):
    """Codec providing j2k compression via Glymur.
    Parameters
    ----------
    psnr : int
        Compression peak signal noise ratio.
    """

    codec_id = "j2k"

    def __init__(self, psnr=50):
        self.psnr = psnr
        assert (self.psnr > 0 and self.psnr <= 100
                and isinstance(self.psnr, int))
        super().__init__()

    def encode(self, buf):
        bufa = ensure_ndarray(buf)
        tmp = tempfile.NamedTemporaryFile()
        buff = glymur.Jp2k(tmp.name, shape=bufa.shape)
        buff._write(bufa, psnr=[30], numres=1)
        f = open(tmp.name, 'rb')
        array = f.read()
        return array

    def decode(self, buf, out=None):
        buf = ensure_bytes(buf)
        if out is not None:
            out = ensure_bytes(out)
        tmp = tempfile.NamedTemporaryFile(delete=False)
        with open(tmp.name, "wb") as fb:
            fb.write(buf)
        jp2 = glymur.Jp2k(tmp.name)
        fullres = jp2[:]
        tiled = fullres
        return ndarray_copy(tiled, out)

register_codec(j2k)