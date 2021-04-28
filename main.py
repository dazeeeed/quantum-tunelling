from utility import Animation, Physics


def main():
    x_min, x_max = -50, 50

    # mass of particle
    # 9.109e-31
    m = 1

    ph = Physics(x_min, x_max, m)
    # barrier - place of the center of the barrier,
    # height of the barrier,
    # width of the barrier,
    # type of the barrier - square or smooth
    ph.make_barrier(8, 3.5, 0.5, 'square')

    model = ph.get_model()

    animation = Animation(model)
    animation.start()


if __name__ == '__main__':
    main()
