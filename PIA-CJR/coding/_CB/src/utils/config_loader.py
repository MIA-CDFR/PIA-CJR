from pathlib import Path
import yaml
import re


def resolve_variables(value, config):

    pattern = re.compile(r"\$\{([^}^{]+)\}")

    if isinstance(value, str):

        matches = pattern.findall(value)

        for match in matches:

            if match in config:
                value = value.replace(
                    f"${{{match}}}",
                    str(config[match])
                )

    return value


def load_config():

    # caminho absoluto até à pasta _CB
    project_root = Path(__file__).resolve().parents[2]

    # caminho absoluto do config.yaml
    config_path = project_root / "configs" / "config.yaml"

    print(f"\nLoading config from:\n{config_path}\n")

    # verificar se existe
    if not config_path.exists():
        raise FileNotFoundError(
            f"\nConfig file not found:\n{config_path}\n"
        )

    # carregar yaml
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # resolver variáveis ${...}
    for key, value in config.items():
        config[key] = resolve_variables(value, config)

    return config