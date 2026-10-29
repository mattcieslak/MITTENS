"""Freeze the legacy Fortran transition-probability coefficients as a golden table.

Run once against the generated Fortran that shipped with MITTENS v1
(``src/singleODF_odf8_ss0_87_am35.f90``, last present in commit 626cc85).
The literals are parsed as float64 from their 25-digit text, i.e. the values
the generator intended, not the float32 values gfortran actually compiled.

Each Fortran subroutine ``<name>_prob`` computes the probability of moving to
the neighbor at integer offset ``RAS_SHIFTS[name]`` expressed in the frame of
the raw DSI Studio odf8 vertex coordinates (which are LPS+).

Output arrays (``legacy_odf8_ss087_am35.npz``):
  vertices : (642, 3) odf8 sphere exactly as stored in mittens/data/odfs.mat
  offsets  : (T, 3) int8 neighbor offset of each term
  chains   : (T, 3) int16 hemisphere direction indices of the hops (-1 padded)
  coefs    : (T,) float64 box-volume coefficient V(sigma, v)
"""
import re
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

RAS_AXES = {"r": (1, 0, 0), "l": (-1, 0, 0), "a": (0, 1, 0),
            "p": (0, -1, 0), "s": (0, 0, 1), "i": (0, 0, -1)}


def parse_terms(fortran_path):
    text = Path(fortran_path).read_text().replace("&\n", "")
    subs = re.findall(r"subroutine (\w+)_prob\((.*?)\)(.*?)end subroutine", text, re.S)
    offsets, chains, coefs = [], [], []
    for name, _, body in subs:
        offset = np.sum([RAS_AXES[ch] for ch in name], axis=0)
        for line in body.splitlines():
            line = line.strip()
            if not line.startswith("P=P+"):
                continue
            coef = float(re.match(r"P=P\+\s*([0-9.]+)", line).group(1))
            first = int(re.search(r"X\((\d+)\)", line).group(1))
            hops = dict((int(a), int(b)) for a, b in re.findall(r"G\((\d+),(\d+)\)", line))
            chain = [first]
            while chain[-1] in hops and len(chain) <= len(hops):
                chain.append(hops[chain[-1]])
            if len(chain) != len(hops) + 1:
                raise ValueError("Could not order hops in: " + line)
            offsets.append(offset)
            chains.append(chain + [-1] * (3 - len(chain)))
            coefs.append(coef)
    return (np.array(offsets, dtype=np.int8), np.array(chains, dtype=np.int16),
            np.array(coefs, dtype=np.float64))


if __name__ == "__main__":
    repo = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[2]
    offsets, chains, coefs = parse_terms(repo / "src" / "singleODF_odf8_ss0_87_am35.f90")
    vertices = loadmat(repo / "mittens" / "data" / "odfs.mat")["odf8_vertices"].T
    out = Path(__file__).with_name("legacy_odf8_ss087_am35.npz")
    np.savez_compressed(out, vertices=vertices.astype(np.float64), offsets=offsets,
                        chains=chains, coefs=coefs)
    print("wrote", out, "terms:", len(coefs))
