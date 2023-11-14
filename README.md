# python-gpmf-metadata

A Python implementation of the [GPMF metadata extractor](https://gopro.github.io/labs/control/metadata/) for GoPro MP4, LRV, 360 and JPG files.

The module has no dependencies and requires only Python 3.9 or newer.

## Usage

In the code:

```python
from gpmf_metadata import Metadata

for four_cc, value in Metadata("tests/fixtures/gopro_video.mp4"):
    print(f"{four_cc}: {value}")
```

Or in the CLI:

```
python gpmf_metadata/metadata.py tests/fixtures/gopro_video.mp4
```

Output:

```
DEVC: 
DVID: 1
DVNM: Global Settings
VERS: [8, 2, 0]
FMWR: H22.01.01.09.23
LINF: LSU2053004401613
CINF: [24, 80, -86, -13, 90, 78, -21, 17, -127, 86, 20, 56, -113, 99, -100, 19]
CASN: C3471324501320
MINF: HERO11 Black
MUID: [-206942184, 300633690, 940856961, 329016207, 0, 0, 0, 0]
CMOD: 12
MTYP: 0
OREN: U
DZOM: N
DZST: 0
SMTR: N
PRTN: Y
PTWB: AUTO
PTSH: MED
PTCL: NATURAL
EXPT: AUTO
PIMX: 0
PIMN: 0
PTEV: 0.0
RATE: 
SROT: 24.0
EISE: Y
EISA: HS AutoBoost
HCTL: Off
AUPT: N
AUDO: AUTO
BROD: 
BRID: 0
PVUL: F
PRJT: GPRO
SOFF: 0
CLKS: 1
CDAT: 0x000000006314CC15
SCTM: 0
PRNA: 1
PRNU: 0
SCAP: N
CDTM: 0
DUST: NO_LIMIT
VRES: [5312, 2988]
VFPS: [30000, 1001]
HSGT: OFF
BITR: STANDARD
MMOD: STEREO
RAMP: 
TZON: 120
CLKC: 0
DZMX: 0.0
CTRL: Pro
PWPR: PERFORMANCE
ORDP: Y
CLDP: Y
PIMD: AUTO
DEVC: 
DVID: FOVL
DVNM: Large FOV
ABSC: 0.0
ZFOV: 141.09014892578125
VFOV: H
MXCF: x1x3x5x7x9x11x13x1y2
MAPX: [1.5805143117904663, -8.166882514953613, 74.5198745727539, -451.500244140625, 1551.292236328125, -2735.542236328125, 1923.1572265625, -0.10860265046358109]
MYCF: y1y3y1x2y1x4
MAPY: [1.023822546005249, -0.10256711393594742, -0.2639929950237274, 0.29792657494544983]
PYCF: r0r1r2r3r4r5r6
POLY: [0.0, 1.8229533433914185, 0.1068095788359642, -0.6631866097450256, 0.35631099343299866, -1.0743799106665078e-13, 4.5314372100566797e-14]
ZMPL: 0.6548827886581421
ARUW: 1.1428571939468384
ARWA: 1.7777777910232544
DEVC: 
DVID: FOVS
DVNM: Small FOV
ABSC: 1.0
ZFOV: 138.60414123535156
VFOV: S
MXCF: x1x3x5
MAPX: [1.2100392580032349, -1.275840163230896, 1.7751845121383667]
MYCF: y1y3y5y1x2y3x2y1x4
MAPY: [0.9364505410194397, 0.4465307891368866, -0.7683315277099609, -0.35740867257118225, 1.1584652662277222, 0.3529347777366638]
PYCF: r0r1r2r3r4r5r6
POLY: [0.0, 1.8229533433914185, 0.1068095788359642, -0.6631866097450256, 0.35631099343299866, -1.0743799106665078e-13, 4.5314372100566797e-14]
ZMPL: 0.6548827886581421
ARUW: 1.3333333730697632
ARWA: 1.7777777910232544
DEVC: 
DVID: HLMT
DVNM: Highlights
```

## Tests

```
python -m unittest tests/tests.py
......
----------------------------------------------------------------------
Ran 6 tests in 0.157s

OK
```

## Compatibility

All GoPro cameras since HERO5 Black (updated: Sept 19, 2023).

## Links

* [GPMF Parser](https://gopro.github.io/gpmf-parser/)
* [GPMF Metadata Extractor](https://gopro.github.io/labs/control/metadata/)
* GoPro unedited [photos](https://www.dpreview.com/sample-galleries/0101586095/gopro-hero-11-black-sample-gallery/7805945349) and [videos](https://mega.nz/folder/YkEEUb7b#ySxIooiC6dsw0bVT9V2jzA) used during the development and testing