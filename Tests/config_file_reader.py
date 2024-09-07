import configparser


def read_config(file_path='config.ini', section=None):
    """
    Universal function to read a configuration file.

    :param file_path: Path to the configuration file (default is 'config.ini').
    :param section: Section name to read. If None, returns all sections.
    :return: Dictionary with keys and values from the configuration file.
    """
    config = configparser.ConfigParser()
    config.read(file_path)

    config_values = {}

    # Getting values from selected section
    if section:
        if section in config:
            for key, value in config.items(section):
                # Reading values boolean and other
                if value.lower() in ['true', 'false']:
                    config_values[key.upper()] = config.getboolean(section, key)
                else:
                    config_values[key.upper()] = value
        else:
            raise ValueError(f"Section '{section}' not found in file {file_path}.")
    else:
        raise ValueError(f"The section was not selected, data was not retrieved from the configuration file.")

    return config_values