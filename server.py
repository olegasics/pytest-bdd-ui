import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import ast
import gherkin

logging.basicConfig(level=logging.DEBUG)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FixtureScanner(ast.NodeVisitor):
    def __init__(self):
        self.fixtures = []
        self.step_definitions = []

    def visit_FunctionDef(self, node):
        has_return = any(isinstance(stmt, ast.Return) for stmt in node.body)
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Name):
                    logging.debug(f"Checking decorator: {decorator.func.id}")
                    if decorator.func.id == "fixture":
                        self.fixtures.append((node.name, has_return))
                        logging.debug(f"Found fixture: {node.name}, returns value: {has_return}")
                    elif decorator.func.id in ["given", "when", "then"]:
                        step_type = decorator.func.id
                        step_text = decorator.args[0].s if decorator.args else ""
                        self.step_definitions.append((step_type, step_text))
                        logging.debug(f"Found step definition: {step_type} - {step_text}")
                        # Associate fixture with step if used in the function
                        for arg in node.args.args:
                            if arg.arg in [fixture[0] for fixture in self.fixtures]:
                                for i, fixture in enumerate(self.fixtures):
                                    if fixture[0] == arg.arg:
                                        self.fixtures[i] = (fixture[0], step_text)
                                        logging.debug(f"Associated fixture {fixture[0]} with step: {step_text}")
                elif isinstance(decorator.func, ast.Attribute):
                    logging.debug(f"Checking attribute decorator: {decorator.func.attr}")
                    if decorator.func.attr == "fixture":
                        self.fixtures.append((node.name, has_return))
                        logging.debug(f"Found fixture: {node.name}, returns value: {has_return}")
            elif isinstance(decorator, ast.Attribute):
                logging.debug(f"Checking attribute decorator: {decorator.attr}")
                if decorator.attr == "fixture":
                    self.fixtures.append((node.name, has_return))
                    logging.debug(f"Found fixture: {node.name}, returns value: {has_return}")
        self.generic_visit(node)

    def scan_file(self, filepath):
        try:
            with open(filepath, "r") as file:
                try:
                    logging.debug(f"Scanning file: {filepath}")
                    node = ast.parse(file.read(), filename=filepath)
                    self.visit(node)
                except (SyntaxError, UnicodeDecodeError) as e:
                    logging.warning(f"Skipping file {filepath} due to parsing error: {e}")
        except Exception as e:
            logging.error(f"Skipping file {filepath} due to error: {e}")

    def scan_directory(self, directory):
        for root, _, files in os.walk(directory):
            # Skip .venv directory
            if '.venv' in root:
                continue
            for file in files:
                if file.endswith(".py"):
                    self.scan_file(os.path.join(root, file))
        return self.fixtures


class FixtureRequest(BaseModel):
    fixtures: dict
    content: str
    directory: str


@app.get("/scan-fixtures")
async def scan_fixtures():
    scanner = FixtureScanner()
    # Use the current working directory as the starting point
    user_app_directory = os.getcwd()
    fixtures = scanner.scan_directory(user_app_directory)

    return {"fixtures": fixtures}


@app.post("/generate-feature")
async def generate_feature(request: FixtureRequest):
    logging.info(f"Received request content: {request.content}")
    try:
        # Use the second element of each tuple for step text
        given_steps = [step[1] for step in request.fixtures.get('given', [])]
        when_steps = [step[1] for step in request.fixtures.get('when', [])]
        then_steps = [step[1] for step in request.fixtures.get('then', [])]

        logging.debug(f"Given steps: {given_steps}")
        logging.debug(f"When steps: {when_steps}")
        logging.debug(f"Then steps: {then_steps}")

        # Generate feature content
        feature_content = (
            "Feature: Generated Scenario\n\n"
            "  Scenario: User configured scenario\n"
            + "\n".join([f"    Given {step}" for step in given_steps])
            + "\n"
            + "\n".join([f"    When {step}" for step in when_steps])
            + "\n"
            + "\n".join([f"    Then {step}" for step in then_steps])
            + "\n"
        )

        feature_path = os.path.join(os.getcwd(), "user_app/generated_scenario.feature")
        logging.info(f"Writing to file: {feature_path}")
        logging.info(f"Feature content to write: {feature_content}")
        with open(feature_path, "w") as feature_file:
            feature_file.write(feature_content)

        return {"message": "Feature file generated successfully!"}
    except Exception as e:
        logging.error(f"Error generating feature file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def analyze_fixtures(fixtures):
    fixture_order = []
    for fixture in fixtures:
        fixture_name = fixture[0]  # Извлечение имени фикстуры
        returns_value = fixture[1]  # Булево значение
        if returns_value:
            fixture_order.append((fixture_name, "target_fixture"))
        else:
            fixture_order.append((fixture_name, None))
    return fixture_order


def parse_feature_file(feature_file):
    with open(feature_file, "r") as file:
        feature_content = file.read()

    parsed_feature = gherkin.parser.Parser().parse(feature_content)
    logging.debug("Parsed feature structure: %s", parsed_feature)

    steps = []
    for scenario in parsed_feature["feature"]["children"]:
        if "scenario" in scenario:
            for step in scenario["scenario"]["steps"]:
                steps.append(step["text"])

    return steps


def parse_feature_file_for_fixtures(feature_file, fixtures, step_definitions):
    with open(feature_file, "r") as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        # Check if the line corresponds to a step that uses a fixture with a return
        for step_type, step_text in step_definitions:
            if step_text in line:
                for fixture_name, has_return in fixtures:
                    if has_return and fixture_name in line:
                        # Add target_fixture to the step
                        line = line.replace(step_text, f"{step_text} [target_fixture={fixture_name}]")
                        break
        modified_lines.append(line)

    # Write the modified lines back to the feature file
    with open(feature_file, "w") as file:
        file.writelines(modified_lines)

    steps = []

    for line in modified_lines:
        line = line.strip()
        if (
            line.startswith("Given")
            or line.startswith("When")
            or line.startswith("Then")
        ):
            steps.append(line)

    return steps


def generate_dynamic_fixture_code(fixture_order):
    code_lines = [
        "import pytest",
        "from pytest_factoryboy import register",
        "import factory",
        "",
        "class DynamicFactory(factory.Factory):",
        "    class Meta:",
        "        model = dict",
        "",
    ]

    added_fixtures = set()

    for fixture, target in fixture_order:
        fixture_name = fixture[0].replace(" ", "_").lower()  # Извлечение имени фикстуры

        if fixture_name not in added_fixtures:
            if target:
                code_lines.append(
                    f"    {fixture_name} = factory.LazyFunction(lambda: '{fixture[0]}')"
                )
            else:
                code_lines.append(
                    f"    {fixture_name} = factory.LazyFunction(lambda: '{fixture[0]}')"
                )
            added_fixtures.add(fixture_name)

    code_lines.append("")
    code_lines.append("register(DynamicFactory, 'dynamic_fixture')")

    return "\n".join(code_lines)


def analyze_fixtures_for_target(steps, fixtures):
    fixture_map = {
        fixture[0]: fixture[1] for fixture in fixtures if isinstance(fixture, tuple)
    }

    target_fixtures = []
    for i, step in enumerate(steps):
        step_name = step.split(" ", 1)[-1]  # Use the last part after splitting
        if step_name in fixture_map and fixture_map[step_name]:
            target_fixtures.append((step_name, True))
        else:
            target_fixtures.append((step_name, False))
    return target_fixtures


@app.post("/generate-tests")
async def generate_test(request: FixtureRequest):
    logging.info(
        "Received request to generate tests with fixtures: %s", request.fixtures
    )
    try:
        feature_file = "user_app/generated_scenario.feature"
        output_file = "user_app/tests/functional/test_generated.py"

        scanner = FixtureScanner()
        scanner.scan_directory(request.directory)
        fixtures = scanner.fixtures
        step_definitions = scanner.step_definitions

        # Read steps from the feature file
        with open(feature_file, 'r') as file:
            feature_content = file.readlines()

        steps = []
        for line in feature_content:
            line = line.strip()
            if line.startswith('Given') or line.startswith('When') or line.startswith('Then'):
                steps.append(line)

        # Use the extracted steps to generate tests
        target_fixtures = analyze_fixtures_for_target(steps, fixtures)
        logging.debug("Target fixtures: %s", target_fixtures)

        # Ensure the directory exists before writing the output file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            f.write("from pytest_bdd import given, when, then\n\n")
            for i, (step_name, has_target) in enumerate(target_fixtures):
                step_type = step_name.split()[0].lower()
                function_name = step_name.replace(' ', '_').replace('[', '').replace(']', '').lower()
                if has_target:
                    f.write(f"@{step_type}('{step_name}', target_fixture='result_{i}')\n")
                    f.write(
                        f"def {function_name}():\n    return 'result_{i}'\n\n"
                    )
                elif i > 0 and target_fixtures[i - 1][1]:
                    previous_function_name = target_fixtures[i - 1][0].replace(' ', '_').replace('[', '').replace(']', '').lower()
                    f.write(f"@{step_type}('{step_name}')\n")
                    f.write(
                        f"def {function_name}(result_{i - 1}):\n    pass\n\n"
                    )
                else:
                    f.write(f"@{step_type}('{step_name}')\n")
                    f.write(
                        f"def {function_name}():\n    pass\n\n"
                    )

        logging.info("Test file generated successfully with dynamic target_fixture.")

        return {"status": "success"}
    except Exception as e:
        logging.error("An unexpected error occurred: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))