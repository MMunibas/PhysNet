from typing import Union
import sys 

sys.path.append('..')

#from unittest import result
from fastapi import FastAPI, Request
from ase import Atoms
import numpy as np
from calculator.NNCalculator import NNCalculator

app = FastAPI()

charges = []
cartesian_coord = []
test_atom = None 
calc = None

@app.get("/")
def read_root():
    return {"Hello ... ": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

@app.get("/contents")
def show_contents():
    out = []
    # read dataset into memory
    data = np.load("../sn2_reactions.npz")

    # print contents to console
    # out += "Contents:\n"
    for key in data.keys():
        d = {"key" : key, "Values" : data[key].shape }
        out.append(d)
    # print()
    return {"Contensts": out}

@app.post("/postCharges")
async def allocateCharges(info : Request):
    req_info = await info.json()
    global charges
    charges = req_info
    print(charges)
    return {
        "status" : "SUCCESS",
        "data" : req_info
    }

@app.post("/postCco")
async def allocateCoord(info : Request):
    req_info = await info.json()
    global cartesian_coord
    cartesian_coord = req_info
    print(cartesian_coord)
    return {
        "status" : "SUCCESS",
        "data" : req_info
    }

@app.get("/getValues")
def showValues():
    global charges, cartesian_coord
    return{
        "Charges" : charges,
        "Cartesian Coordinates" : cartesian_coord
    }

@app.get("/initAtoms")
def initAtom():
    global test_atom, charges, cartesian_coord, calc
    test_atom = Atoms(numbers=charges, positions=cartesian_coord)
    use_electrostatic = True
    use_dispersion = True
    calc = NNCalculator(checkpoint='../tests/checkpoint/best_model.ckpt', atoms=test_atom, use_dispersion=use_dispersion, use_electrostatic=use_electrostatic)
    return{
        "status" : "Success",
    }

@app.get("/getCharges")
def calc_charges():
    global calc, test_atom 
    return{
        "Charges": calc.get_charges(test_atom).tolist()
    }

@app.get("/getForces")
def calc_forces():
    global calc, test_atom 
    return{
        "Forces": calc.get_forces(test_atom).tolist()
    }
    
@app.get("/getPe")
def calc_pe():
    global calc, test_atom 
    return{
        "PE": calc.get_potential_energy(test_atom).tolist()
    }

# test_atom = Atoms(numbers=data["Z"][idx,:data["N"][idx]], positions=data["R"][idx,:data["N"][idx],:])