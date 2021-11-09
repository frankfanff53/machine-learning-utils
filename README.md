# Machine Learning Utils Package 
Welcome to this package repo, this package is designed for supporting learning of Introduction to Machine Learning at Imperial College London.

## Main Project Structure

```
.
├── ./ml_utils
├── ./test
├── ./scripts
├── ./requirements.txt
└── ./setup.py
```

The `scripts` folder contains the installation scripts of venv in setting up the developing environment. We would discuss the issues about setting up in the following **setup** part.

The `test` folder is the placeholder for our private tests of our implementations. The tests use the `pytest` framework to do unit testing. More information would be covered in the **testing** part.

## Setting up your virtual environment for development
For MacOS users, use the following command to run the script given to create the venv and install dependencies:
```bash
./script/install_venv venv
```
and for Windows users, use an alternative script to do all the jobs above:
```bash
./script/install_venv_win venv
```
The script would not only set up a virtual environment, but also the required python packages for this package, including `numpy`,  `pandas`, `pytest` and `pytorch` etc.

**Note:** For Linux users and machines, currently I found the installation of pytorch on Linux OS is not working properly, as it would always makes the CI-CD pipeline on shared runner broke. I am trying to solve this quesiton and hope it could be fixed soon.

## Testing
Unlike the previous coursework, this time we use the `pytest` framework to do unit testing. To run all tests, you could run this command:
```python
pytest test/
```
where pytest as been installed to your virtual environment.

And we could use:
```python
pytest test/test_demo.py
```
to check the implemntation of a specific part.

To run a specific test, you could specify the testing function name after your testing file, like:
```python
# Just append the reference name to the end of testing file
# do not append parameters after function names
pytest test/test_demo.py::your-testing-function 
```

That is all of the information you need at the very beginning.