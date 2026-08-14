from __future__ import annotations

from crt import CRTSystem


def main() -> None:
    system = CRTSystem((5, 7, 11))
    x, y = 9, 9
    ex = system.encode(x)
    ey = system.encode(y)

    add_latent = system.add(ex, ey)
    mul_latent = system.mul(ex, ey)

    print(f"Moduli: {system.moduli}")
    print(f"Capacity: {system.capacity}")
    print(f"E({x}) = {ex}")
    print(f"E({y}) = {ey}")
    print()
    print(f"E({x}) ⊕ E({y}) = {add_latent}")
    print(f"E({(x + y) % system.capacity}) = {system.encode((x + y) % system.capacity)}")
    print(f"Addition exact: {add_latent == system.encode((x + y) % system.capacity)}")
    print()
    print(f"E({x}) ⊗ E({y}) = {mul_latent}")
    print(f"E({(x * y) % system.capacity}) = {system.encode((x * y) % system.capacity)}")
    print(f"Multiplication exact: {mul_latent == system.encode((x * y) % system.capacity)}")


if __name__ == "__main__":
    main()
