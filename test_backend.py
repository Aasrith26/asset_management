import subprocess


def generate_requirements():
    try:
        # Run `pip freeze` and capture the output
        result = subprocess.run(
            ["pip", "freeze"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )

        # Write the output to requirements.txt
        with open("requirements1.txt", "w") as f:
            f.write(result.stdout)

        print("requirements.txt has been generated successfully.")

    except subprocess.CalledProcessError as e:
        print("Error occurred while generating requirements.txt")
        print(e.stderr)


if __name__ == "__main__":
    generate_requirements()
