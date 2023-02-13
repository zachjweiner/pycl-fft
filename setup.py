#!/usr/bin/env python

import sys
from pathlib import Path
from setuptools import setup


def find_git_revision(tree_root):
    tree_root = Path(tree_root).resolve()

    if not tree_root.joinpath(".git").exists():
        return None

    from subprocess import run, PIPE, STDOUT
    result = run(["git", "rev-parse", "HEAD"], shell=False,
                 stdin=PIPE, stdout=PIPE, stderr=STDOUT, close_fds=True,
                 cwd=tree_root)

    git_rev = result.stdout
    git_rev = git_rev.decode()
    git_rev = git_rev.rstrip()

    assert result.returncode is not None
    if result.returncode != 0:
        from warnings import warn
        warn("unable to find git revision")
        return None

    return git_rev


def write_git_revision(package_name):
    dn = Path(__file__).parent
    git_rev = find_git_revision(dn)
    text = 'GIT_REVISION = "%s"\n' % git_rev
    (dn / package_name / "_git_rev.py").write_text(text)


write_git_revision("pycl_fft")


def get_config():
    include_dirs = []
    library_dirs = []
    extra_compile_args = []
    extra_link_args = None

    if "darwin" in sys.platform:
        libraries = None
        extra_link_args = ["-Wl,-framework,OpenCL"]
    elif "linux" in sys.platform:
        libraries = ["OpenCL"]
    else:
        raise NotImplementedError("Windows")

    include_dirs.append("VkFFT/vkFFT/")

    import os
    conda_dir = os.environ.get("CONDA_PREFIX")
    clfft_dir = os.environ.get("CLFFT_PATH", conda_dir)

    if clfft_dir is not None:
        clfft_dir = Path(clfft_dir)
        if (clfft_dir / "lib/libclFFT.so").is_file():
            libraries.append("clFFT")
            library_dirs.append(str(clfft_dir / "lib"))
            include_dirs.append(str(clfft_dir / "include"))

    return {
        "libraries": libraries,
        "extra_link_args": extra_link_args,
        "include_dirs": include_dirs,
        "library_dirs": library_dirs,
        "extra_compile_args": extra_compile_args
    }


def main():
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    conf = get_config()
    ext_modules = [
        Pybind11Extension("pycl_fft._vkfft", ["src/wrap-vkfft.cpp"], **conf)
    ]
    if "clFFT" in conf["libraries"]:
        ext_modules.append(
            Pybind11Extension(
                "pycl_fft._clfft", ["src/wrap-clfft.cpp"], **conf)
        )

    setup(
        cmdclass={
            "build_ext": build_ext,
        },
        include_package_data=True,
        ext_modules=ext_modules,
    )


if __name__ == "__main__":
    main()
