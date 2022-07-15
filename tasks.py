from invoke import task


@task
def build_docs(context):
    context.run("sphinx-apidoc -o ./docs ./rxnutils")
    context.run("sphinx-build -M html ./docs ./docs/build")


@task
def run_tests(context):
    cmd = (
        "pytest --black --mccabe "
        "--cov rxnutils --cov-branch --cov-report html:coverage --cov-report xml "
        "-vv tests/"
    )
    context.run(cmd)


@task
def run_linting(context):
    print("Running pylint...")
    context.run("pylint rxnutils")