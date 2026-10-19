import numpy as np
import matplotlib.pyplot as plt


def frontera(E, mu, n=600):
    """
    Frontera de la región admisible en el plano (r, r punto).
    De (1+mu) rdot^2 + 2 r (mu-1) = 2E se despeja r en función de rdot:
        r(rdot) = [E - (1+mu) rdot^2 / 2] / (mu - 1)
    """
    rdot_tope = np.sqrt(2 * E / (1 + mu))      # r = 0
    rdot = np.linspace(-rdot_tope, rdot_tope, n)
    r_front = (E - 0.5 * (1 + mu) * rdot**2) / (mu - 1)
    return rdot, r_front, rdot_tope


def graficar_region(E=1.0, mu=2.8, archivo="actividad 8.png"):
    rdot, r_front, rdot_tope = frontera(E, mu)
    r_tope = E / (mu - 1)                      # rdot = 0

    relleno = "#c4a7e7"   # lila claro
    borde = "#4a148c"     # morado oscuro

    fig, ax = plt.subplots(figsize=(8, 5.5))

    ax.fill_betweenx(rdot, 0, r_front, color=relleno, alpha=0.8,
                     label="Región admisible")
    ax.plot(r_front, rdot, color=borde, lw=2)

    ax.plot([0, 0, r_tope], [rdot_tope, -rdot_tope, 0], "o", color=borde)
    ax.annotate(rf"$r_{{max}}={r_tope:.3f}$", (r_tope, 0),
                textcoords="offset points", xytext=(-10, 12), ha="right")
    ax.annotate(rf"$|\dot r|_{{max}}={rdot_tope:.3f}$", (0, rdot_tope),
                textcoords="offset points", xytext=(10, -4))

    ax.axhline(0, color="k", lw=0.7)
    ax.axvline(0, color="k", lw=0.7)
    ax.set(xlim=(-0.02, 1.1 * r_tope), ylim=(-1.15 * rdot_tope, 1.15 * rdot_tope),
           xlabel=r"$r$", ylabel=r"$\dot r$",
           title=rf"Región admisible: $E={E}$, $\mu={mu}$")
    ax.grid(alpha=0.3, ls="--")
    ax.legend(loc="upper right")
    fig.tight_layout()

    fig.savefig(archivo, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    return r_tope, rdot_tope


r_tope, rdot_tope = graficar_region(E=1.0, mu=2.8)
print(f"r_max     = {r_tope:.3f}")
print(f"|rdot|max = {rdot_tope:.3f}")