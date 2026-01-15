## Build API Docs

1. Clone the `puppet-warp` repository.
2. Change directory to `puppet-warp/docs`.
3. Install the documentation requirements.
4. Ensure your virtual environment is activated.
5. Create a `build` directory.
6. Run the following command:

```bash
sphinx-build -W -b html -c ./source -d ./build/doctrees ./source ./build
```

> ⚠️ Make sure the path separators are valid for your operating system.

---

## Generate API Docs (Developers Only)

1. Change directory to the docs folder:

```bash
cd /path/to/docs
```

2. Activate your virtual environment.
3. Remove the old API docs folder.
4. Generate the API documentation:

```bash
sphinx-apidoc ../src/pwarp -o ./source/api -f
```

5. Ensure `source/api/pwarp.rst` contains the following at the end of the file:

```rst
.. automodule:: pwarp
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
```

6. Build the documentation:

```bash
sphinx-build -W -b html -c ./source -d ./build/doctrees ./source ./build -v
```