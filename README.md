# Template

这是一个通用的 Python 项目模板仓库，预配置了代码质量检查和格式化工具，旨在帮助开发者快速启动符合规范的 Python 项目。

## 功能特性

本项目集成了 `pre-commit` 工具，在提交代码时自动执行以下检查和修复：

*   **代码风格检查**: 使用 `flake8` 进行 Python 代码风格检查。
*   **导入排序**: 使用 `isort` 自动规范和排序 Python 导入语句。
*   **代码格式化**: 使用 `yapf` 自动格式化 Python 代码，保持风格统一。
*   **常规检查**:
    *   `trailing-whitespace`: 删除行尾多余空格。
    *   `check-yaml`: 检查 YAML 文件语法。
    *   `end-of-file-fixer`: 确保文件以换行符结尾。
    *   `requirements-txt-fixer`: 规范 `requirements.txt` 文件。
    *   `double-quote-string-fixer`: 统一字符串引号风格。
    *   `check-merge-conflict`: 检查是否存在未解决的合并冲突。
    *   `fix-encoding-pragma`: 移除不必要的编码声明。
    *   `mixed-line-ending`: 统一行尾结束符为 LF。

## 快速开始

1.  **克隆仓库**

    ```bash
    git clone <repository_url>
    cd Template
    ```

2.  **安装依赖**

    你需要先安装 `pre-commit`：

    ```bash
    pip install pre-commit
    ```

3.  **安装 Pre-commit Hooks**

    在项目根目录下运行：

    ```bash
    pre-commit install
    ```

    安装完成后，每次执行 `git commit` 时，上述配置的钩子将会自动运行，确保提交的代码符合质量标准。

## 配置说明

*   **`.pre-commit-config.yaml`**: 定义了所有的 git hooks 配置。本项目使用了 OpenMMLab 在 Gitee 的镜像源，以确保在国内环境下的下载速度。
*   **`.flake8`**: `flake8` 的配置文件，定义了忽略规则、最大行长度 (79) 等。

## 许可证

LICENSE
