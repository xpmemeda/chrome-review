# - 2020年12月，Red Hat 把工作重心从 CentOS Linux 转移到 CentOS Stream ，前者是 RHEL 下游，后者是 RHEL 下游，更好的利用社区力量，最后版本的 CentOS Linux 是 CentOS 8
# - DockerHub 上面的 centos 账户已经废弃。
# - CentOS Stream 基础镜像下载链接： https://www.centos.org/
FROM quay.io/centos/centos:10 AS s0

RUN dnf update -y
# |Package Name                             |Description                                                                                                                                                                                    |
# |-                                        |-                                                                                                                                                                                              |
# |zsh                                      |A powerful and interactive shell with advanced features such as auto-completion and customization.                 |
# |wget                                     |A command-line utility for downloading files from the web via HTTP, HTTPS, and FTP protocols.                      |
# |bzip2                                    |A compression tool for creating and extracting .bz2 files.                                                         |
# |gcc                                      |The GNU Compiler Collection, a standard compiler for C, C++, and other languages.                                  |
# |make                                     |A build automation tool that automatically builds executable programs and libraries from source code.              |
# |which                                    |A utility to locate a command and display its full path.                                                           |
# |perl                                     |A high-level, general-purpose programming language, often used for scripting and system administration.            |
# |git                                      |A distributed version control system for tracking changes in source code during software development.              |
# |git-lfs                                  |Git Large File Storage, an extension for Git to handle large files efficiently.                                    |
# |vim                                      |A highly configurable text editor built to enable efficient text editing.                                          |
# |procps                                   |A set of utilities for monitoring and managing system processes, such as ps, top, and free.                        |
# |dnf-plugins-core                         |A set of core plugins that extend the functionality of the DNF package manager.                                    |
# |pkg-config                               |A tool that provides compiler and linker flags for libraries, simplifying the build process.                       |
# |gmp-devel                                |Development files for the GNU Multiple Precision Arithmetic Library, used for high-precision arithmetic.           |
# |mpfr-devel                               |Development files for the Multiple Precision Floating-Point Reliable Library.                                      |
# |libmpc-devel                             |Development files for the Multiple Precision Complex Arithmetic Library, used for complex number calculations.     |
# |libxcrypt-compat-4.4.36-9.el10.x86_64    |Compatibility library for traditional and extended crypt password hashing algorithms,                              |
# |                                         |providing legacy API (libcrypt.so.1) support for applications that require the old crypt interface.                |
RUN dnf install zsh wget bzip2 gcc make which perl gmp-devel mpfr-devel libmpc-devel git git-lfs dnf-plugins-core pkg-config vim procps libxcrypt-compat-4.4.36-9.el10.x86_64 -y
# dnf builddep
#   - 解析指定软件包的 .spec 文件（通常位于 SRPM 中）。
#   - 提取构建依赖（BuildRequires 部分）。
#   - 安装这些依赖包。
RUN dnf builddep --enablerepo=crb python3 -y
# dnf group install
#   - 安装指定的软件包组。
#   - "Development Tools"：软件包组的名称，通常包含编译和开发所需的工具（如 GCC、Make、Git 等）。
RUN dnf group install "Development Tools" -y

# time zone: CST
ENV TZ=Asia/Shanghai
RUN dnf install tzdata -y && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# torch==2.9.0
RUN python -m venv ~/envs/cpython-3.12.13/torch-2.9.0
RUN source ~/envs/cpython-3.12.13/torch-2.9.0/bin/activate && pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0

# tini
RUN wget https://github.com/krallin/tini/releases/download/v0.19.0/tini -O /usr/bin/tini && chmod +x /usr/bin/tini

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sleep", "infinity"]
