from flask import Flask, request, make_response
import pickle
import time
import argparse
from datetime import datetime
import os


app = Flask(__name__)



class flask:
    c=0
    state_list = {}
    state = None
    clients=0
    ids = 0
    closed=False
    
        # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%y%m%d")
    # print("date and time =", dt_string)	
    i = 0
    while 1:
        checkpoint_path = 'saved_model/global_model_%s_%i.pkl' %(dt_string, i)
        if os.path.exists(checkpoint_path):
            i+=1
        else:
            break
            
    def __init__(self,args):
        flask.clients = int(args.IEDs)
        flask.ids=int(args.IEDs)

def add(flask):
    while flask.c<flask.ids:
        time.sleep(0.20)
        if flask.closed:
            return None
        if flask.state is not None:
            break
    if flask.state is None:
        print("average weights")
        for key in flask.state_list[0]:
#             flask.state_list[0][key]=sum(flask.state_list[x][key] for x in range(1,flask.ids))/flask.ids
            for x in range(1,flask.ids):
                flask.state_list[0][key]=(flask.state_list[0][key]+flask.state_list[x][key])#/(flask.ids)
            flask.state_list[0][key]= flask.state_list[0][key]/flask.ids
        flask.state = flask.state_list[0]

@app.route('/close', methods=['POST'])
def close():
    flask.closed=True
    return 'closing',200        

@app.route('/getid', methods=['POST'])
def getid():
    if flask.clients!=0:
        ids=flask.clients
        flask.clients-=1
        # flask.state_list[flask.ids] = None
        print("connection from IED {ids} to server ".format(ids=str(flask.clients)))
        return str(flask.clients),200
    else:
        return "None",400

    


@app.route('/upload', methods=['POST'])
def upload():
    while flask.state is not None:
        time.sleep(0.20)

    
    file = pickle.loads(request.data)
    print("*****received weights from IED {ids} to server*****".format(ids=file[0]))
    # print(file[0],"aaaaa")
    if file[0]  in flask.state_list:
        print(f"already exists id {msg}".format(file[0]))
        return f"already exists id {file[0]}", 400

    flask.state_list[file[0]]=file[1]
    flask.c+=1
    add(flask)
    if flask.closed:
        flask.c-=1
        flask.state_list = {}
        flask.state = None
        response=make_response(pickle.dumps("closed"))
        response.headers['Content-Type'] = 'application/octet-stream'
        if flask.c==0:
            print("closing from server side")
        return response,400
    # print(file)
    print("**********sending back global model to IEDs, running on {post}**********".format(post=request.headers['Host']))
    response=make_response(pickle.dumps(flask.state))
    response.headers['Content-Type'] = 'application/octet-stream'
    flask.c-=1
    # print(flask.state)

    
    if flask.c==0:
        if args.save_model:
            with open(flask.checkpoint_path, 'wb') as file: 
                pickle.dump(flask.state, file) 
        flask.state_list = {}
        flask.state = None
    return response, 200 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--IEDs",help="number of local IED particiapting", default=2)
    parser.add_argument("--save_model",help="True for saving the global model after each iteration",default=True)
    args=parser.parse_args()
    # pritn(args)
    obj=flask(args)
    app.debug = True
    app.run(host="0.0.0.0")