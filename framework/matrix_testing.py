import os
import time
import uuid
import warnings
from itertools import chain, islice, repeat
from types import LambdaType
from itertools import chain, islice, repeat

import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
from evalPol import eval3, eval5, eval9, eval17, sastre8

# TODO: Apply these to 0.001 see the error to check why it does not work with the given deg 3 approxes.
# 17, 17, 17
co17 = [

    (

        [

            [8.1900628402201772e00, -1.1341497882285287e01],

            [5.2695286609514156e00, -1.1355755098785176e01, -8.5575587792319769e00],

            [2.9941920143595552e-01, 3.9368736424870771e-01, -1.3629994852556693e-01, 8.7543792821015454e-01],

        ],

        [

            [5.7224200607981572e00, -1.3349570510955273e01],

            [2.4027229678283074e01, -6.9033542160046899e00, -8.1581776809768751e00],

            [3.4280673478743980e-02, -5.9554503046777407e-01, -1.4997426806130245e00, 2.8797501228862719e00],

        ],

        [

            1.2650651343045305e-01,

            3.6010993010002004e-03,

            1.8975934474204015e00,

            -1.0000000000000000e00,

            1.0000000000000000e00,

        ],

    ),

    (

        [

            [1.1144074461067206e01, -6.9396828758687201e00],

            [7.9990669472586662e00, -1.5203810732953235e01, -1.1864829667404932e01],

            [4.8726421145938135e00, -5.8118918475287673e00, -4.1495523653680859e00, 3.7670050455967421e00],

        ],

        [

            [2.7920643049709448e00, -1.2173180152033812e01],

            [1.8330227610086052e01, -1.2595988939658326e01, -8.5818359752227487e00],

            [5.4573434354147443e-01, -6.5204669665263626e-01, -4.8864015526669952e-01, -3.6134936920750149e-01],

        ],

        [

            -6.7015134954514274e-03,

            1.3207317978063923e-02,

            -1.3271454360899630e00,

            -1.0000000000000000e00,

            1.0000000000000000e00,

        ],

    ),

    (

        [

            [1.6299250683234664e01, -1.0916580743419150e01],

            [3.7913032455099862e00, -1.2091853643971568e01, -8.4497129859472579e00],

            [2.0443860705498897e00, -2.3035547895774053e00, -3.0226551737973932e00, -1.7898310627611815e00],

        ],

        [

            [5.2460060407784832e00, -1.6298612492955382e01],

            [2.4110518765100437e01, -1.0443034129627778e01, -7.0686358901823443e00],

            [-7.5667962333679650e00, 4.7505407528854944e00, 2.0656187790382869e00, 1.8643556133192400e00],

        ],

        [

            -5.3633761697932086e-03,

            2.0793412852751460e-02,

            -5.2926244835196729e-02,

            -1.0000000000000000e00,

            1.0000000000000000e00,

        ],

    ),

    (

        [

            [1.6031574106559976e01, -1.1147498488902956e01],

            [4.2026958242656027e00, -1.1950450600727466e01, -8.1313541853831222e00],

            [2.5803801639056574e00, -1.6476836162133901e00, -2.8026154399513548e00, -1.9348904425589695e00],

        ],

        [

            [5.1105010106341799e00, -1.6447337553819647e01],

            [2.4323150509918314e01, -9.7183194558286310e00, -7.3093303735788435e00],

            [-7.5994104159191522e00, 4.7819666281751774e00, 1.6589010640978155e00, 1.7980217613035476e00],

        ],

        [

            -3.6470399492116688e-03,

            1.9560705832703747e-02,

            -3.5247930092353254e-02,

            -1.0000000000000000e00,

            1.0000000000000000e00,

        ]

    )

]

# 17, 17, 9
co9 = [
    (
        [
            [8.19006284e00, -1.13414979e01],
            [5.26952866e00, -1.13557551e01, -8.55755878e00],
            [2.99419201e-01, 3.93687364e-01, -1.36299949e-01, 8.75437928e-01],
        ],
        [
            [5.72242006e00, -1.33495705e01],
            [2.40272297e01, -6.90335422e00, -8.15817768e00],
            [3.42806735e-02, -5.95545030e-01, -1.49974268e00, 2.87975012e00],
        ],
        [
            1.26506513e-01,
            3.60109930e-03,
            1.89759345e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.11440745e01, -6.93968288e00],
            [7.99906695e00, -1.52038107e01, -1.18648297e01],
            [4.87264211e00, -5.81189185e00, -4.14955237e00, 3.76700505e00],
        ],
        [
            [2.79206430e00, -1.21731802e01],
            [1.83302276e01, -1.25959889e01, -8.58183598e00],
            [5.45734344e-01, -6.52046697e-01, -4.88640155e-01, -3.61349369e-01],
        ],
        [
            -6.70151350e-03,
            1.32073180e-02,
            -1.32714544e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [[8.18628571, -7.34995052], [0.08376457, -4.48494194, -4.06476615]],
        [[1.33726249, -0.96757271], [-6.2348802, 0.38265358, 0.0163027]],
        [-1.48495011, -0.04376982, 1.0, -1.0],
    ),
]

# 17,17,5
co5 = [
    (
        [
            [8.19006284e00, -1.13414979e01],
            [5.26952866e00, -1.13557551e01, -8.55755878e00],
            [2.99419201e-01, 3.93687364e-01, -1.36299949e-01, 8.75437928e-01],
        ],
        [
            [5.72242006e00, -1.33495705e01],
            [2.40272297e01, -6.90335422e00, -8.15817768e00],
            [3.42806735e-02, -5.95545030e-01, -1.49974268e00, 2.87975012e00],
        ],
        [
            1.26506513e-01,
            3.60109930e-03,
            1.89759345e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.11440745e01, -6.93968288e00],
            [7.99906695e00, -1.52038107e01, -1.18648297e01],
            [4.87264211e00, -5.81189185e00, -4.14955237e00, 3.76700505e00],
        ],
        [
            [2.79206430e00, -1.21731802e01],
            [1.83302276e01, -1.25959889e01, -8.58183598e00],
            [5.45734344e-01, -6.52046697e-01, -4.88640155e-01, -3.61349369e-01],
        ],
        [
            -6.70151350e-03,
            1.32073180e-02,
            -1.32714544e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (2.64972986, -1.93611987, 0.43470742),
]

co3 = [
    (
        [
            [8.19006284e00, -1.13414979e01],
            [5.26952866e00, -1.13557551e01, -8.55755878e00],
            [2.99419201e-01, 3.93687364e-01, -1.36299949e-01, 8.75437928e-01],
        ],
        [
            [5.72242006e00, -1.33495705e01],
            [2.40272297e01, -6.90335422e00, -8.15817768e00],
            [3.42806735e-02, -5.95545030e-01, -1.49974268e00, 2.87975012e00],
        ],
        [
            1.26506513e-01,
            3.60109930e-03,
            1.89759345e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    (
        [
            [1.11440745e01, -6.93968288e00],
            [7.99906695e00, -1.52038107e01, -1.18648297e01],
            [4.87264211e00, -5.81189185e00, -4.14955237e00, 3.76700505e00],
        ],
        [
            [2.79206430e00, -1.21731802e01],
            [1.83302276e01, -1.25959889e01, -8.58183598e00],
            [5.45734344e-01, -6.52046697e-01, -4.88640155e-01, -3.61349369e-01],
        ],
        [
            -6.70151350e-03,
            1.32073180e-02,
            -1.32714544e00,
            -1.00000000e00,
            1.00000000e00,
        ],
    ),
    [1.97717358, -0.55454523],
    (1.66512672, -0.51373244),
]


co17high = [

    (

        [

            [8.1900628402201772e00, -1.1341497882285287e01],

            [5.2695286609514156e00, -1.1355755098785176e01, -8.5575587792319769e00],

            [2.9941920143595552e-01, 3.9368736424870771e-01, -1.3629994852556693e-01, 8.7543792821015454e-01],

        ],

        [

            [5.7224200607981572e00, -1.3349570510955273e01],

            [2.4027229678283074e01, -6.9033542160046899e00, -8.1581776809768751e00],

            [3.4280673478743980e-02, -5.9554503046777407e-01, -1.4997426806130245e00, 2.8797501228862719e00],

        ],

        [

            1.2650651343045305e-01,

            3.6010993010002004e-03,

            1.8975934474204015e00,

            -1.0000000000000000e00,

            1.0000000000000000e00,

        ],

    ),

    (

        [

            [1.1144074461067206e01, -6.9396828758687201e00],

            [7.9990669472586662e00, -1.5203810732953235e01, -1.1864829667404932e01],

            [4.8726421145938135e00, -5.8118918475287673e00, -4.1495523653680859e00, 3.7670050455967421e00],

        ],

        [

            [2.7920643049709448e00, -1.2173180152033812e01],

            [1.8330227610086052e01, -1.2595988939658326e01, -8.5818359752227487e00],

            [5.4573434354147443e-01, -6.5204669665263626e-01, -4.8864015526669952e-01, -3.6134936920750149e-01],

        ],

        [

            -6.7015134954514274e-03,

            1.3207317978063923e-02,

            -1.3271454360899630e00,

            -1.0000000000000000e00,

            1.0000000000000000e00,

        ],

    ),

    (

        [

            [1.6299250683234664e01, -1.0916580743419150e01],

            [3.7913032455099862e00, -1.2091853643971568e01, -8.4497129859472579e00],

            [2.0443860705498897e00, -2.3035547895774053e00, -3.0226551737973932e00, -1.7898310627611815e00],

        ],

        [

            [5.2460060407784832e00, -1.6298612492955382e01],

            [2.4110518765100437e01, -1.0443034129627778e01, -7.0686358901823443e00],

            [-7.5667962333679650e00, 4.7505407528854944e00, 2.0656187790382869e00, 1.8643556133192400e00],

        ],

        [

            -5.3633761697932086e-03,

            2.0793412852751460e-02,

            -5.2926244835196729e-02,

            -1.0000000000000000e00,

            1.0000000000000000e00,

        ],

    ),

    (

        [

            [1.6031574106559976e01, -1.1147498488902956e01],

            [4.2026958242656027e00, -1.1950450600727466e01, -8.1313541853831222e00],

            [2.5803801639056574e00, -1.6476836162133901e00, -2.8026154399513548e00, -1.9348904425589695e00],

        ],

        [

            [5.1105010106341799e00, -1.6447337553819647e01],

            [2.4323150509918314e01, -9.7183194558286310e00, -7.3093303735788435e00],

            [-7.5994104159191522e00, 4.7819666281751774e00, 1.6589010640978155e00, 1.7980217613035476e00],

        ],

        [

            -3.6470399492116688e-03,

            1.9560705832703747e-02,

            -3.5247930092353254e-02,

            -1.0000000000000000e00,

            1.0000000000000000e00,

        ]

    )

]

# safety factor for numerical stability (but exclude last polynomial)

for i in range(len(co3) - 2):
    for j in range(len(co3[i][2])):
        co3[i][2][j] /= 1.01 ** (j+2)
co3[2][0] /= 1.01
co3[2][1] /= 1.01**3


for i in range(len(co5) - 1):
    for j in range(len(co5[i][2])):
        co5[i][2][j] /= 1.01 ** (j+2)

for i in range(len(co9)):
    for j in range(len(co9[i][2])):
        co9[i][2][j] /= 1.01 ** (j+2)

for i in range(len(co17)):
    for j in range(len(co17[i][2])):
        co17[i][2][j] /= 1.01 ** (j+2)

"""coeffs_list = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]
] + [coeffs_list[-1]]"""


@torch.compile
def MachPolarStep(G: torch.Tensor, t: int, m: int) -> torch.Tensor:
    if m == 3:
        co = co17high
    elif m == 2:
        co = co9
    elif m == 1:
        co = co5
    else:
        co = co3
    assert G.ndim >= 2
    X = G.double()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs

    n = X.shape[0]

    if m == 1:
        A = X @ X.mT
        B = co[2][1] * A + co[2][2] * A @ A
        X = co[2][0] * X + B @ X
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X
    elif m == 0:
        A = X @ X.mT
        X = co[2][0] * X + co[2][1] * A @ X
        if G.size(-2) > G.size(-1):
            X = X.mT
        return X

    A = co[t][0]
    B = co[t][1]
    c = co[t][2]
    t += 1
    out = torch.zeros(m + 2, n, n, dtype=X.dtype, device=X.device)
    out[0] = torch.eye(n, dtype=X.dtype, device=X.device)  # "1" as identity matrix
    out[1] = X @ X.mT
    for i in range(m):
        out1 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
        out2 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
        for j in range(len(A[i])):
            out1 = out1 + A[i][j] * out[j]
            out2 = out2 + B[i][j] * out[j]

        out[i + 2] = c[i] * (out1 @ out2)

    X = torch.sum(out, dim=0) @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def MachPolar9(G: torch.Tensor, steps: int) -> torch.Tensor:
    co = co9
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    n = X.shape[0]
    t = 0

    m_list = [3, 3, 2]

    for m in m_list:
        A = co[t][0]
        B = co[t][1]
        c = co[t][2]
        t += 1
        out = torch.zeros(m + 2, n, n, dtype=X.dtype, device=X.device)
        out[0] = torch.eye(n, dtype=X.dtype, device=X.device)  # "1" as identity matrix
        out[1] = X @ X.mT
        for i in range(m):
            out1 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            out2 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            for j in range(len(A[i])):
                out1 = out1 + A[i][j] * out[j]
                out2 = out2 + B[i][j] * out[j]

            out[i + 2] = c[i] * (out1 @ out2)

        X = torch.sum(out, dim=0) @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def MachPolar5(G: torch.Tensor, steps: int) -> torch.Tensor:
    co = co5
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    n = X.shape[0]
    t = 0

    m_list = [3, 3]

    for m in m_list:
        A = co[t][0]
        B = co[t][1]
        c = co[t][2]
        t += 1
        out = torch.zeros(m + 2, n, n, dtype=X.dtype, device=X.device)
        out[0] = torch.eye(n, dtype=X.dtype, device=X.device)  # "1" as identity matrix
        out[1] = X @ X.mT
        for i in range(m):
            out1 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            out2 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            for j in range(len(A[i])):
                out1 = out1 + A[i][j] * out[j]
                out2 = out2 + B[i][j] * out[j]

            out[i + 2] = c[i] * (out1 @ out2)

        X = torch.sum(out, dim=0) @ X

    A = X @ X.mT
    B = co[2][1] * A + co[2][2] * A @ A
    X = co[2][0] * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def MachPolar17(G: torch.Tensor, steps: int) -> torch.Tensor:
    co = co17
    assert G.ndim >= 2
    X = G.bfloat16()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    n = X.shape[0]
    t = 0

    m_list = [3, 3, 3]

    for m in m_list:
        A = co[t][0]
        B = co[t][1]
        c = co[t][2]
        t += 1
        out = torch.zeros(m + 2, n, n, dtype=X.dtype, device=X.device)
        out[0] = torch.eye(n, dtype=X.dtype, device=X.device)  # "1" as identity matrix
        out[1] = X @ X.mT
        for i in range(m):
            out1 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            out2 = torch.zeros(n, n, dtype=X.dtype, device=X.device)
            for j in range(len(A[i])):
                out1 = out1 + A[i][j] * out[j]
                out2 = out2 + B[i][j] * out[j]

            out[i + 2] = c[i] * (out1 @ out2)

        X = torch.sum(out, dim=0) @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X



coeffs_list = [
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375),  # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
coeffs_list = [
    (a / 1.01, b / 1.01**3, c / 1.01**5) for (a, b, c) in coeffs_list[:-1]
] + [coeffs_list[-1]]


coeffs_normal17 = [[ 2.5991361115901309e+01, -7.9282849900909127e+02,
        9.8024936239615126e+03, -5.8080876194237993e+04,
        1.8700484279994160e+05, -3.4562337876708852e+05,
        3.6622936261970218e+05, -2.0675900872931961e+05,
        4.8195375794365755e+04], [ 1.1378064053415830e+01, -8.9067683509732007e+01,
        2.8260446948165577e+02, -4.2971169171262915e+02,
        3.5505727658121458e+02, -1.6840315700680617e+02,
        4.5793288458872674e+01, -6.6345987905669892e+00,
        3.9687811202301393e-01], [   4.817415986776974  ,  -21.93954931921574   ,
         62.52382523542986   , -100.98701337692943   ,
         96.18014483833946   ,  -55.02848715728586   ,
         18.558329275079622  ,   -3.395558630210992  ,
          0.25972273107363353], [  3.383746341645896  ,  -9.263227903156402  ,
        19.952043171898392  , -29.20942885786782   ,
        29.078167048195965  , -19.47356156921225   ,
         8.423211818813874  ,  -2.1308972758999114 ,
         0.23994722558226453]]





@torch.compile
def PolarExpress(G: torch.Tensor, a: float, b: float, c: float) -> torch.Tensor:
    assert G.ndim >= 2
    X = G.double()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs

    A = X @ X.mT
    B = b * A + c * A @ A
    X = a * X + B @ X  # X <- aX + bX^3 + cX^5
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

@torch.compile
def PolarTest(G: torch.Tensor, coeffs) -> torch.Tensor:
    # TODO: accumulate in 32 bult mult in 16
    assert G.ndim >= 2
    X = G.double()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    for i in range(len(coeffs) - 1):
        coeffs[i] /= 1.01 ** (2 * i + 1)
    a, b, c, d, e, f, g, h, i = coeffs
    A = X @ X.mT
    B = (
        b * A
        + c * A @ A
        + d * A @ A @ A
        + e * A @ A @ A @ A
        + f * A @ A @ A @ A @ A
        + g * A @ A @ A @ A @ A @ A
        + h * A @ A @ A @ A @ A @ A @ A
        + i * A @ A @ A @ A @ A @ A @ A @ A
    )
    X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


def synthetic_matrix(m, n, smallest=1e-3, largest=1e-3, device="cpu"):
    k = min(m, n)

    # Random orthogonal U
    U, _ = torch.linalg.qr(torch.randn(m, m, device=device))

    # Random orthogonal V
    V, _ = torch.linalg.qr(torch.randn(n, n, device=device))

    # Log-spaced singular values
    s = torch.logspace(
        torch.log10(torch.tensor(largest)),
        torch.log10(torch.tensor(smallest)),
        steps=k,
        device=device,
    )

    # Rectangular Sigma
    Sigma = torch.zeros(m, n, device=device)
    Sigma[:k, :k] = torch.diag(s)

    A = U @ Sigma @ V.T
    return A, s


def synthetic_plots():
    A = torch.tensor([[0.001]])

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polarFactor = U @ Vh



    A17 = A.clone()
    APE = A.clone()
    # print(torch.linalg.matrix_norm(MachPolar17(A, 3) - polarFactor, ord=2))

    A17 = A17 / (A17.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    APE = APE / (APE.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    spec17 = [1]
    specPE = [1]
    m_list=[3,3,3]
    for i in range(3):
        A17 = MachPolarStep(A17, i, m_list[i])
        spec17.append(torch.linalg.matrix_norm(A17 - polarFactor, ord=2)/ torch.linalg.matrix_norm(polarFactor, ord=2))

    for i in range(5):
        a = coeffs_list[i][0]
        b = coeffs_list[i][1]
        c = coeffs_list[i][2]
        APE = PolarExpress(APE, a, b, c)
        specPE.append(torch.linalg.matrix_norm(APE - polarFactor, ord=2)/ torch.linalg.matrix_norm(polarFactor, ord=2))
    print(spec17)

    plt.plot([0, 5, 10, 15], spec17, label="Approximation with degrees 17, 17, 17")
    plt.plot([0, 3, 6, 9, 12, 15], specPE, label="Polar Express")
    plt.xlabel("Matrix-Matrix Multiplications")
    plt.ylabel("Relative Frobenius Error")
    plt.legend()
    plt.show()


from17to5 = [[2.65705496/ 1.01, -1.97367367/1.01**3,  0.45223007/1.01**5], [ 1.95502978 / 1.01, -1.33631062/1.01**3,  0.38373029/1.01**5], [ 1.89538159 / 1.01, -1.29109485 / 1.01**3,  0.39571404 / 1.01**5], [ 1.87499975 / 1.01, -1.24999949 / 1.01**3,  0.37499975 / 1.01**5]]
def real_plots():
    """Error when using a matrix from the GPT training."""
    A = torch.load("h3_c_attn_grads.pt", map_location="cpu")
    if not torch.is_tensor(A):
        # in case it's saved as a state_dict / dict of tensors
        raise ValueError("Loaded object is not a tensor — inspect its structure first")
    A = A.double()
    if A.ndim > 2:
        # flatten any leading batch/head dims down to a single 2D matrix
        A = A.reshape(-1, A.shape[-1])

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polarFactor = U @ Vh

    A17 = A.clone()
    APE = A.clone()

    A17 = A17 / (A17.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    APE = APE / (APE.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    spec17 = [1]
    specPE = [1]
    specNormal = [1]
    m_list=[3,3,3,3]

    print("Mach!")
    for i in range(4):
        A17 = MachPolarStep(A17, i, m_list[i])
        spec17.append(torch.linalg.matrix_norm(A17 - polarFactor, ord="fro")/ torch.linalg.matrix_norm(polarFactor, ord="fro"))

    for i in range(0):
        if i >= 4:
            a = 1.875
            b = -1.25
            c = 0.375
        else:
            a = from17to5[i][0]
            b = from17to5[i][1]
            c = from17to5[i][2]
        A17 = PolarExpress(A17, a, b, c)
        spec17.append(torch.linalg.matrix_norm(A17 - polarFactor, ord="fro") / torch.linalg.matrix_norm(polarFactor, ord="fro"))

    print("PE!")
    for i in range(10):
        if i >= 7:
            a = 1.875
            b = -1.25
            c = 0.375
        else:
            a = coeffs_list[i][0]
            b = coeffs_list[i][1]
            c = coeffs_list[i][2]
        APE = PolarExpress(APE, a, b, c)
        specPE.append(torch.linalg.matrix_norm(APE - polarFactor, ord="fro") / torch.linalg.matrix_norm(polarFactor, ord="fro"))

    A17 = A.mT.clone()
    APE = A.mT.clone()
    coeffs = coeffs_normal17
    print("17 Naive!")
    for i in range(len(coeffs)):
        print(i)
        APE = APE*0
        for j in range(len(coeffs[i])):
            print("j")
            print(j)

            APE = APE + coeffs[i][j] * torch.linalg.matrix_power(A17@A17.mT, j) @ A17
        A17 = APE.clone()
        specNormal.append(torch.linalg.matrix_norm(A17.mT - polarFactor, ord="fro") / torch.linalg.matrix_norm(polarFactor, ord="fro"))
        print(specNormal[-1])

    plt.plot([0, 5, 10, 15, 20], spec17, label="Approximation with degrees 17, 17, 17 Polset")
    plt.plot([0, 5, 10, 15, 20], specNormal, label="Approximation with degrees 17, 17, 17")
    plt.plot([3*i for i in range(11)], specPE, label="Polar Express")
    plt.xlabel("Matrix-Matrix Multiplications")
    plt.ylabel("Relative Frobenius Error")
    plt.legend()
    plt.show()


def main():
    real_plots()



main()
