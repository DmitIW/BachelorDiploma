import subprocess
import sys

path_to_tex_file = "/home/dmitri/PycharmProjects/BachelorDiploma/thesis/presentation/presentation.tex"
executable_builder = "/usr/bin/xelatex"


def main() -> None:
    cmd: str = "{0} {1}".format(
        executable_builder,
        path_to_tex_file
    )
    _ = subprocess.run(
        args=[cmd],
        shell=True,
        stdout=sys.stdout, stderr=sys.stderr,
        encoding='ascii', input='R'
    )


if __name__ == "__main__":
    main()
