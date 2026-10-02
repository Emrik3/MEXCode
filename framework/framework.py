import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
from evalPol import eval3, eval5, eval9, eval17, sastre8


coeffs_dir = "coeffs/"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 12
plt.rcParams["axes.linewidth"] = 1.5
plt.rc("text", usetex=False)
plt.rc("legend", fontsize=10)


def p(c, x):
    out = 0
    for i in range(len(c)):
        out += c[i] * x ** (2 * i + 1)
    return out


# Derivative of polynomial
def pp(c, x):
    out = 0
    for i in range(len(c)):
        out += c[i] * (2 * i + 1) * x ** (2 * i)
    return out


# Second derivative of polynomial
def ppp(c, x):
    out = 0
    for i in range(1, len(c)):
        out += c[i] * (2 * i + 1) * (2 * i) * x ** (2 * i - 1)
    return out


def roots(c, l, u, steps=100000):
    """Find roots of odd poynomial with coefficients c."""
    xs = np.linspace(l, u, steps)
    guesses = []

    prev_x = xs[0]
    prev_val = pp(c, prev_x)

    for x in xs[1:]:
        curr_val = pp(c, x)

        # Check for sign change
        if prev_val * curr_val < 0:
            guesses.append((prev_x + x) / 2)
        prev_x = x
        prev_val = curr_val

    return guesses




def odd_remez(q, l, u, tol):
    """Remez algorithm for approximation of the sign function of any degree."""
    x = np.zeros(q + 2)
    f = np.ones(q + 2)
    n = q + 2
    # Calculate initial guess of points as Chebyshev points
    for i in range(n):
        x[i] = 0.5 * (l + u) + 0.5 * (u - l) * np.cos((i) * np.pi / (n))
    x = np.array(sorted(x))
    old_E = np.inf
    E = 1000

    while np.abs(old_E - E) > tol:
        old_E = E
        A = np.zeros((n, n))
        for j in range(n):
            for i in range(n - 1):
                A[j, i] = x[j] ** (2 * i + 1)
        A[:, -1] = (-1) ** np.arange(n)
        try:
            c = np.linalg.solve(A, f)
        except np.linalg.LinAlgError:
            if q == 2:
                return [1.875, -1.25, 0.375, 0]
            else:
                raise np.linalg.LinAlgError("Singular Matrix")

        x_new = []
        coeffs_for_roots = derivative_coeffs(c[:-1])
        root_guess = np.roots(coeffs_for_roots)
        candidates = []
        for r in root_guess:
            if np.isreal(r):
                r = r.real
                if r > 0:
                    candidates.append(
                        r
                    )  # TODO: might have to change basis for high degree polynomials cause cant find all the roots.
        # Makes sence since we get like 0.001**41 and stuff like that.

        for guess in candidates:  # If they are too close we might have problems
            x_new.append(newton_pol(guess, c[:-1], tol))

        # Always include endpoints
        x_new = [l] + x_new + [u]

        # Sort for consistency
        x_new = np.array(sorted(x_new))

        # Make sure all unique points were found
        if len(x_new) == len(set(x_new)):
            x = np.array(x_new)
        else:
            print("Error: Could not find all points")
            break
        E = c[-1]

    return c


def derivative_coeffs(c):
    """
    Construct coefficients of p'(x) for an odd polynomial
    p(x) = sum c[i] x^(2i+1)

    Returns coefficients in descending powers for np.roots
    """
    q = len(c) - 1
    coeffs = np.zeros(2 * q + 1)

    for i, ci in enumerate(c):
        power = 2 * i
        coeffs[2 * q - power] = ci * (2 * i + 1)

    return coeffs


def newton_pol(x, c, tol):
    """Newton's method the refine root search"""
    err = 100.0
    while err > tol:
        x_new = x - pp(c, x) / ppp(c, x)

        err = np.abs(x_new - x)
        x = x_new

    return x


def plot_pol(c, l, u):
    x = np.linspace(l, u, 100000)
    plt.plot(x, p(c, x))
    plt.plot(x, 1 + 0 * x)


def get_all_coeffs_different_degrees(q_list, T, l=0.001):
    """Main function using Remez to get the coefficents for total composition of polynomials of different degrees."""
    cushion = 0.02407327424182761 # Set cushion to 0 if it is not wished to be used.
    # cushion = 0
    u = 1
    all_coeffs = []
    eps = 1e-10
    llist = [l]
    k = 1
    # TODO: Possible to find the exact approximant of degree 17 using pade?

    for i in range(T):
        if 1-l <= 1e-9:
            if q_list[i] == 8:
                const_coeffs = [109395, -291720, 612612, -875160, 850850, -556920, 235620, -58344, 6435]
                const_coeffs = [c / 32768 for c in const_coeffs]
                all_coeffs.append(const_coeffs)
                continue
            else:
                all_coeffs.append( [1.875, -1.25, 0.375])
                continue
        q = q_list[i]
        c = odd_remez(
            q, max(l, cushion * u), u, 1e-8
        )  # Make  more exact?
        if cushion * u > l:
            pl = p(c[:-1], l)
            pu = p(c[:-1], u)
            rescalar = 2 / (pl + pu)
            for i in range(len(c[:-1])):
                c[i] *= rescalar

        #for i in range(len(c) - 1):
        #    c[i] /= (1.01) ** (2 * i + 1)

        l = p(c[:-1], l)
        llist.append(l)
        x = np.linspace(l, u, 1000)

        u = 2-l

        all_coeffs.append(c[:-1])
        if len(llist) > 1:
            print(1 - l)
            # print((1 - llist[np.max(p(c[:-1], x))k]) / (1 - llist[k - 1]) ** (q + 1))
        k += 1

    return all_coeffs


@torch.compile
def PolarExpress(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    """Degree 5 five times."""
    assert G.ndim >= 2
    X = G.double()  # This is used for testing, should use bfloat16 in machine learning setting.
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs

    for a, b, c in coeffs_list:
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X  # X <- aX + bX ˆ3 + cX ˆ5
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


@torch.compile
def NewPolarExpress(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    """Total evaluation of the polynomial composition using."""
    # TODO: accumulate in 32 bult mult in 16
    assert G.ndim >= 2
    X = G.bfloat16() # This is used for testing, should use bfloat16 in machine learning setting.
    if G.size(-2) > G.size(-1):
        X = X.mT

    for c in coeffs_list:
        if len(c) == 2:
            X = eval3(X, c)
        elif len(c) == 3:
            X = eval5(X, c)
        elif len(c) == 5:
            X = eval9(X, c)
        elif len(c) == 9:
            X = eval17(X, c)
        else:
            raise NotImplementedError("This mult not impl!")
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X.to(G.dtype)


"""Compare answer."""
@torch.compile
def PolarTest(G: torch.Tensor, steps: int, coeffs_list) -> torch.Tensor:
    # TODO: accumulate in 32 bult mult in 16
    assert G.ndim >= 2
    X = G.bfloat16() # This is used for testing, should use bfloat16 in machine learning setting.
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    for c in coeffs_list:
        for i in range(len(c) - 1):
            c[i] /= 1.01 ** (2 * i + 1)
    for a, b, c, d, e, f, g, h, i in coeffs_list:
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

    # TODO: Some issues with the convergence of newton when the tol is too high, for large polynomials it does not converge.


colormap = {2: "#6298D2", 8: "#004791"}


def test_approximation(q, l=0.001):
    """Test function, get coefficents of all polynomials in composition and plot the resulting approximation."""
    T = len(q)
    coeffs17 = get_all_coeffs_different_degrees(q, T, l)
    print(coeffs17)
    np.save(coeffs_dir + "coeffs.npy", np.array(coeffs17, dtype=object))
    x_plt = np.linspace(0, 1, 10000)

    x = np.linspace(0, 1, 10000)
    tot_degree = 1
    for i in range(T):
        x = p(coeffs17[i], x)
        tot_degree *= 2 * q[i] + 1
        l = p(coeffs17[i], l)

    real_plots(coeffs17, q)
    """plt.plot(
        x_plt,
        x,
        linewidth=1.5,
        label=r"Total degree $= {}$, $d = {}$".format(
            int(tot_degree), [int(v) for v in 2 * np.array(q) + 1]
        ),
        # color=colormap.get(q[0], None),
    )
    plt.legend(fontsize=10)"""


def test_polar():
    """Test accuracy in matrix setting"""
    q = [8, 8, 8]  
    qPE = [2, 2, 2, 2, 2]
    T = len(q)
    TPE = len(qPE)
    coeffs17 = get_all_coeffs_different_degrees(q, T)
    coeffsPE = get_all_coeffs_different_degrees(qPE, TPE)

    A = torch.abs(torch.randn(500, 70))  
    A = A / (A.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polarFactor = U @ Vh

    sastreCoeffs = []

    for i in range(len(coeffs17)):
        if len(coeffs17[i]) == 9:
            sastreCoeffs.append(sastre8(coeffs17[i]))
        else:
            sastreCoeffs.append(coeffs17[i])

    polarFactorNew = NewPolarExpress(A, T, sastreCoeffs)

    polarFactorPE = PolarExpress(A, TPE, coeffsPE)

    diffPE = polarFactor - polarFactorPE
    diffNew = polarFactor - polarFactorNew

    errPE = torch.linalg.matrix_norm(diffPE, ord=2)
    errNew = torch.linalg.matrix_norm(diffNew, ord=2)

    print(f"Polar Express error: {errPE}")
    print(f"New Express error: {errNew}")


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

    ), 

    (

    [

        [1.6031574105251433e+01, -1.1147498491607324e+01],

        [4.2026958242813279e+00, -1.1950450600711740e+01, -8.1313541839234826e+00],

        [2.5803801639737705e+00, -1.6476836161452770e+00, -2.8026154399376000e+00, -1.9348904438201786e+00],

    ],

    [

        [5.1105010125580437e+00, -1.6447337555126030e+01],

        [2.4323150509696994e+01, -9.7183194560499491e+00, -7.3093303735811146e+00],

        [-7.5994104159694267e+00, 4.7819666281249029e+00, 1.6589010640876634e+00, 1.7980217614247784e+00],

    ],

    [

        -3.6470377160762795e-03,

        1.9560699740462543e-02,

        -3.5247939607092903e-02,

        -1.0000000000000000e+00,

        1.0000000000000000e+00,

    ]

),

]


@torch.compile
def MachPolarStep(G: torch.Tensor, t: int, m: int) -> torch.Tensor:
    co = co17
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


coeffs_listPE = [
(8.28721201814563, -23.595886519098837, 17.300387312530933),
(4.107059111542203, -2.9478499167379106, 0.5448431082926601),
(3.9486908534822946, -2.908902115962949, 0.5518191394370137),
(3.3184196573706015, -2.488488024314874, 0.51004894012372),
(2.300652019954817, -1.6689039845747493, 0.4188073119525673),
(1.891301407787398, -1.2679958271945868, 0.37680408948524835),
(1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
(1.875, -1.25, 0.375), # subsequent coeffs equal this numerically
]
# safety factor for numerical stability (but exclude last polynomial)
coeffs_listPE = [(a / 1.01, b / 1.01**3, c / 1.01**5)
for (a, b, c) in coeffs_listPE[:-1]] + [coeffs_listPE[-1]]

def newton_schulz_step(G):
    assert G.ndim >= 2
    X = G.double()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs

    X = 1.5 * X - 0.5 * (X @ X.mT) @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def jordan_step(G):
    assert G.ndim >= 2
    X = G.double()  # for speed
    if G.size(-2) > G.size(-1):
        X = X.mT  # this reduces FLOPs

    A = (X @ X.mT) 
    X = 3.4445 * X - 4.7750 * A @ X +2.0315 * (A @ A) @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def save_curve(name, xs, ys):
    with open(f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        for x, y in zip(xs, ys):
            w.writerow([float(x), float(y)])   # float() also converts 0-dim torch tensors

def real_plots(coeffs, q_list):
    """Plot relative Frobenius error of a gradient matrix from machine learning setting."""
    A = torch.load("h3_c_attn_grads.pt", map_location="cpu")
    if not torch.is_tensor(A):
        # in case it's saved as a state_dict / dict of tensors
        raise ValueError("Loaded object is not a tensor — inspect its structure first")
    A = A.double()
    if A.ndim > 2:
        # flatten any leading batch/head dims down to a single 2D matrix
        A = A.reshape(-1, A.shape[-1])

    """for i in range(len(co17) - 2):
        for j in range(len(co17[i][2])):
            co17[i][2][j] /= 1.01 ** (j + 2)"""

    s = torch.linalg.svdvals(A)

    print("shape:", A.shape)
    print("sigma max:", s.max().item())
    print("sigma min:", s.min().item())
    print("condition:", (s.max() / s.min()).item())

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polarFactor = U @ Vh


    APE = A.clone()
    ANS = A.clone()
    Ajordan = A.clone()
    APS = A.clone()

    APE = APE / (APE.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    APS = APS / (APS.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    ANS = ANS / (ANS.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)
    Ajordan = Ajordan / (Ajordan.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-7)

    s = torch.linalg.svdvals(APE)
    print("Normalized:")
    print("shape:", APE.shape)
    print("sigma max:", s.max().item())
    print("sigma min:", s.min().item())
    print("condition:", (s.max() / s.min()).item())

    specPE = [1]
    specPS = [1]
    multsPE = [0]
    multsjordan = [0]
    multsPS = [0]

    multsNS = [0]
    specNS = [1]
    specjordan = [1]

    for i in range(10):
        multsPE.append(3*(i+1))
        if i >= 7:
            i = 7
        APE = PolarExpress(APE, 1, [coeffs_listPE[i]])
        specPE.append(torch.linalg.matrix_norm(APE - polarFactor, ord="fro") / torch.linalg.matrix_norm(polarFactor, ord="fro"))# / torch.linalg.matrix_norm(polarFactor, ord="fro")

    for i in range(10):
        multsjordan.append(3*(i+1))
        Ajordan = jordan_step(Ajordan)
        specjordan.append(torch.linalg.matrix_norm(Ajordan - polarFactor, ord="fro") / torch.linalg.matrix_norm(polarFactor, ord="fro"))# / torch.linalg.matrix_norm(polarFactor, ord="fro")


    for i in range(15):
        multsNS.append(2*(i+1))
        ANS = newton_schulz_step(ANS)
        specNS.append(torch.linalg.matrix_norm(ANS - polarFactor, ord="fro") / torch.linalg.matrix_norm(polarFactor, ord="fro"))# / torch.linalg.matrix_norm(polarFactor, ord="fro")
        

    for i in range(6):
        multsPS.append(5*(i+1))
        if i >= 4:
            i = 4
        APS = MachPolarStep(APS, i, 3)
        specPS.append(torch.linalg.matrix_norm(APS - polarFactor, ord="fro") / torch.linalg.matrix_norm(polarFactor, ord="fro")) # / torch.linalg.matrix_norm(polarFactor, ord="fro")

    print(f"Final error PE: {specPE[-1]}")
    print(f"Final error new: {specPS[-1]}")

    q_to_mult = {1: 2, 2: 3, 4: 4, 8: 5, 12: 6}


    mults = [0]
    for q in q_list:
        mults.append(mults[-1] + q_to_mult[q])

    if q_list[0] == 2:
        lab = "Polar Express"
    else:
        lab = "Proposed New Method"
    plt.plot(
        multsPE,
        specPE,
        linewidth=3,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label=lab,
    )

    plt.plot(
            multsjordan,
            specjordan,
            linewidth=3,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Jordan",
        )

    plt.plot(
            multsPS,
            specPS,
            linewidth=3,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Proposed New Method",
        )
    plt.plot(
                multsNS,
                specNS,
                linewidth=3,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="Newton-Schulz",
            )


    plt.xlabel("Matrix-Matrix Multiplications", fontsize=14)
    plt.ylabel("Relative Frobenius Error", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, alpha=0.25)
    plt.ylim(0, 1.05)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(frameon=False, fontsize=11)
    plt.tight_layout()
    save_curve("gradient_frobenius_errorNS", multsNS, specNS)
    save_curve("gradient_frobenius_errorPE", multsPE, specPE)
    save_curve("gradient_frobenius_errorPS", multsPS, specPS)

import torch
import matplotlib.pyplot as plt


def generate_log_spectrum_matrix(m, n, sigma_min=1e-6, sigma_max=1.0):
    """
    Generate an m x n matrix

        A = U @ diag(S) @ V^T

    where the singular values are evenly spaced on a logarithmic
    scale between sigma_min and sigma_max, and U,V are random
    orthogonal matrices.
    """
    k = min(m, n)

    # Random orthogonal singular vectors
    U, _ = torch.linalg.qr(torch.randn(m, k, dtype=torch.double))
    V, _ = torch.linalg.qr(torch.randn(n, k, dtype=torch.double))

    # Logarithmically spaced singular values
    S = torch.logspace(
        torch.log10(torch.tensor(sigma_min, dtype=torch.double)),
        torch.log10(torch.tensor(sigma_max, dtype=torch.double)),
        k,
        dtype=torch.double,
    )

    # A = U diag(S) V^T
    A = (U * S) @ V.T

    return A


def synthetic_plots(coeffs, q_list):
    """
    Plot relative Frobenius error for a synthetic matrix whose
    singular values are logarithmically spaced from 1e-6 to 1.
    """

    # Same dimensions as h3_c_attn_grads.pt
    m = 6912
    n = 768
    l = 1e-3

    A = generate_log_spectrum_matrix(
        m,
        n,
        sigma_min=l,
        sigma_max=1.0,
    )

    # ---------------------------------------------------------
    # Inspect spectrum
    # ---------------------------------------------------------

    s = torch.linalg.svdvals(A)

    print("shape:", A.shape)
    print("sigma max:", s.max().item())
    print("sigma min:", s.min().item())
    print("condition:", (s.max() / s.min()).item())

    # ---------------------------------------------------------
    # Reference polar factor
    # ---------------------------------------------------------

    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    polarFactor = U @ Vh

    k = 2 # 30 * k is the total number of mults per method

    # ---------------------------------------------------------
    # Polar Express
    # ---------------------------------------------------------

    APE = A.clone()

    

    s = torch.linalg.svdvals(APE)

    print("After normalization:")
    print("shape:", A.shape)
    print("sigma max:", s.max().item())
    print("sigma min:", s.min().item())
    print("condition:", (s.max() / s.min()).item())

    specPE = [1.0]
    multsPE = [0]

    for i in range(10*k):

        multsPE.append(3 * (i+1))

        coeff_idx = min(i, 7)

        APE = PolarExpress(
            APE,
            1,
            [coeffs_listPE[coeff_idx]]
        )

        error = (
            torch.linalg.matrix_norm(
                APE - polarFactor,
                ord=2
            )
            
        )

        specPE.append(error.item())


    # Newton schulz
    ANS = A.clone()

    specNS = [1.0]
    multsNS = [0]

    for i in range(15*k):

        multsNS.append(2 * (i+1))

        coeff_idx = min(i, 7)

        ANS = newton_schulz_step(
            ANS
        )

        error = (
            torch.linalg.matrix_norm(
                ANS - polarFactor,
                ord=2
            )
            
        )

        specNS.append(error.item())


    # ---------------------------------------------------------
    # Proposed method
    # ---------------------------------------------------------

    APS = A.clone()

 



    specPS = [1.0]
    multsPS = [0]

    for i in range(6*k):

        multsPS.append(5 * (i+1))

        coeff_idx = min(i, 4)

        APS = MachPolarStep(
            APS,
            coeff_idx,
            3
        )

        error = (
            torch.linalg.matrix_norm(
                APS - polarFactor,
                ord=2
            )
           
        )

        specPS.append(error.item())



    specjordan = [1.0]
    multsjordan = [0]
    Ajordan = A.clone()

    for i in range(10*k):

        multsjordan.append(3 * (i+1))


        Ajordan = jordan_step(Ajordan)

        error = (
            torch.linalg.matrix_norm(
                Ajordan - polarFactor,
                ord=2
            )
            
        )

        specjordan.append(error.item())



    print(f"Final error PE:  {specPE[-1]:.6e}")
    print(f"Final error new: {specPS[-1]:.6e}")

    bound_predictionPE = [1]
    old_bound_predictionPE = [1]
    bound_predictionPS = [1]
    old_bound_predictionPS = [1]
    bound_predictionNS = [1]
    old_bound_predictionNS = [1]

  
    for i in range(6):
        bound_predictionPS.append((1-l)**(9**(i+1)))
        old_bound_predictionPS.append((1-l**2)**(9**(i+1)))

    for i in range(10):
            bound_predictionPE.append((1-l)**(3**(i+1)))
            old_bound_predictionPE.append((1-l**2)**(3**(i+1)))

    for i in range(15):
            bound_predictionNS.append((1-l)**(2**(i+1)))
            old_bound_predictionNS.append((1-l**2)**(2**(i+1)))

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 5))

    plt.semilogy(
        multsPE,
        specPE,
        linewidth=3,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Polar Express",
    )

    plt.semilogy(
            multsjordan,
            specjordan,
            linewidth=3,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Jordan",
        )

    plt.semilogy(
        multsPS,
        specPS,
        linewidth=3,
        marker="o",
        markersize=7,
        markeredgecolor="white",
        markeredgewidth=1.5,
        label="Proposed New Method",
    )

    plt.semilogy(
            multsNS,
            specNS,
            linewidth=3,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Newton-Schulz",
        )

    """plt.plot(
            multsPE,
            bound_predictionPE,
            linewidth=3,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="New Bound Polar Express",
        )"""
    """plt.plot(
                multsPS,
                bound_predictionPS,
                linewidth=3,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="New Bound New Method",
            )"""

    """plt.plot(
            multsPE,
            old_bound_predictionPE,
            linewidth=3,
            marker="o",
            markersize=7,
            markeredgecolor="white",
            markeredgewidth=1.5,
            label="Previous Bound Polar Express",
        )"""
    """plt.plot(
                multsNS,
                old_bound_predictionNS,
                linewidth=3,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="Previous Bound NS",
            )
    plt.plot(
                multsNS,
                bound_predictionNS,
                linewidth=3,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="New Bound NS",
            )"""
    """plt.plot(
                multsPS,
                old_bound_predictionPS,
                linewidth=3,
                marker="o",
                markersize=7,
                markeredgecolor="white",
                markeredgewidth=1.5,
                label="Previous Bound New Method",
            )"""

    

    plt.xlabel(
        "Matrix-Matrix Multiplications",
        fontsize=14
    )

    plt.ylabel(
        "Spectral Error",
        fontsize=14
    )

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.grid(True, alpha=0.25)

    plt.ylim(1e-3, 1.05)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(
        frameon=False,
        fontsize=11
    )

    plt.tight_layout()

    """save_curve("PE_vals_1e-3", multsPE, specPE)
    save_curve("PE_old_bound_1e-3", multsPE, old_bound_predictionPE)
    save_curve("PE_new_bound_1e-3", multsPE, bound_predictionPE)
"""
    plt.savefig(
        "synthetic_log_spectrum_all.pdf",
        format="pdf",
        bbox_inches="tight"
    )

    plt.show()



def approxs():
    np.set_printoptions(
        threshold=np.inf,
            precision=17,
            suppress=False
    )
    plt.figure(figsize=(6.2, 4.0), dpi=150)

    test_approximation([2,2,2,2,2,2, 2,2,2,2])
    #test_approximation([8,8,8,8, 8,8])

    plt.savefig("relfroberrorbfloat.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def main():
    #approxs()
    synthetic_plots(0,0)
    plt.show()
    #   approxs()


if __name__ == "__main__":
    main()
