import requests
import numpy as np
from LSBSteg import encode
from riddle_solvers import riddle_solvers
import pickle


def loaded_model(pth_model):
    with open(pth_model, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


model = loaded_model('arima_model.pkl')

api_base_url = "http://16.171.171.147:5000"
#team_id="bVUrA0A"

def init_fox(team_id: str):
    """
    • Endpoint: /fox/start
    • Method: POST
    • Parameters:
    – teamId (string): The ID of the team participating in the game.
    • Response:
    – msg (string): The secret message.
    – carrier image (array): The carrier image to use, presented as a NumPy array.
    • Example Request:
    {
    ”teamId”: ”team123”
    }
    • Example Response:
    {
    ”msg”: ”This is the secret message.”,
    ”carrier image”: [[0.2 0.4 0.6] [0.3 0.5 0.7], [0.1 0.8 0.9]]
    }
    """

    url = api_base_url+"/fox/start"
    data = {
        "teamId": team_id
    }

    res = requests.post(url, json=data)

    if res.ok:
        res = res.json()
        real_message = res["msg"]
        carrier_image = res["carrier_image"]
        return real_message, np.array(carrier_image)
    else:
        print("Error in start! status code:", res.status_code)
    '''
    In this fucntion you need to hit to the endpoint to start the game as a fox with your team id.
    If a sucessful response is returned, you will recive back the message that you can break into chunkcs
      and the carrier image that you will encode the chunk in it.
    '''

def generate_message_array(message: str, image_carrier: np.array):  
    '''
    In this function you will need to create your own startegy. That includes:
        1. How you are going to split the real message into chunkcs
        2. Include any fake chunks
        3. Decide what 3 chuncks you will send in each turn in the 3 channels & what is their entities (F,R,E)
        4. Encode each chunck in the image carrier  
    '''
    # simple algo that sends the message in one chunk
    fake = ["Dellete", 'Deloite']
    encoded_message = encode(image_carrier.copy(), message)
    enc_fake1 = encode(image_carrier.copy(), fake[0])
    enc_fake2 = encode(image_carrier.copy(), fake[1])
    return [encoded_message.tolist(), enc_fake1.tolist(), enc_fake2.tolist()], ['R', 'F', 'F']



def get_riddle(team_id: str, riddle_id: str):
    '''
    In this function you will hit the api end point that requests the type of riddle you want to solve.
    use the riddle id to request the specific riddle.
    Note that: 
        1. Once you requested a riddle you cannot request it again per game. 
        2. Each riddle has a timeout if you didnot reply with your answer it will be considered as a wrong answer.
        3. You cannot request several riddles at a time, so requesting a new riddle without answering the old one
          will allow you to answer only the new riddle and you will have no access again to the old riddle. 
    '''
    """
    Endpoint: /fox/get-riddle
    • Method: POST
    • Parameters:
    – teamId (string): The ID of the team participating in the game.
    – riddleId (string): The ID of the riddle type requested, as specified in the riddles
    documentation. (e.g., cv easy).
    • Description: This API is used to request a riddle for the fox to solve.
    • Response:
    – test case : A test case for the requested riddle - the format of which depends
    on the riddle as specified in the riddle details documented.
    """
    url = api_base_url+"/fox/get-riddle"
    data = {
        "teamId": team_id,
        "riddleId": riddle_id
    }

    res = requests.post(url, json=data)

    if res.ok:
        res = res.json()
        return res["test_case"]
    else:
        print("Error in get-riddle! status code:", res.status_code)

    


def solve_riddle(team_id: str, solution: str):
    '''
    In this function you will solve the riddle that you have requested. 
    You will hit the API end point that submits your answer.
    Use te riddle_solvers.py to implement the logic of each riddle.
    '''
    """
    3.3 Solve Riddle
    • Endpoint: /fox/solve-riddle
    • Parameters:
    – teamId (string): The ID of the team participating in the game.
    – solution (string): The solution to the riddle in the format expected according
    to the riddle details.
    • Description: This API is used to submit an answer to the riddle. You only have one
    attempt to solve each riddle per game.
    • Response:
    – budget increase: The amount the budget has increased.
    – total budget: The current total budget.
    – status: Indicating success or failure of the solution.
    • Example Request:
    {
    ”teamId”: ”team123”,
    ”solution”: ”The solution to the riddle”
    }
    • Example Response:
    {
    ”budget_increase”: 100,
    ”total_budget”: 1000,
    ”status”: ”success”
    """
    url = api_base_url+"/fox/solve-riddle"
    data = {
        "teamId": team_id,
        "solution": solution
    }

    res = requests.post(url, json=data)

    if res.ok:
        res = res.json()
        inc = res["budget_increase"]
        tot = res["total_budget"]
        status = res["status"]
        return inc, tot, status
    else:
        print("Error in Solve_riddle! status code:", res.status_code)
        return None, None, None


def send_message(team_id, messages, message_entities=['F', 'E', 'R']):
    '''
    Use this function to call the api end point to send one chunk of the message. 
    You will need to send the message (images) in each of the 3 channels along with their entites.
    Refer to the API documentation to know more about what needs to be send in this api call. 
    '''
    """• Endpoint: /fox/send-message
        • Method: POST
        • Parameters:
        – teamId (string): The ID of the team participating in the game.
        – messages (array): An array of three images representing the messages that will
        be sent after being encoded - the images should be sent as NumPy arrays that
        are converted to a list using NumPy’s tolist() method..
        – message entities (array): An array of three characters representing the validity
        of each message (R for real, F for fake, E for empty).
        • Description: This API is used to send the messages and their corresponding validity
        to the Parrot.
        • Response:
        – status (string): success or failure of sending the message.
        • Example Request:
        {
        ”teamId”: ”team123”,
        ”messages”: [image1, image2, image3],
        ”message entities”: [”R”, ”F”, ”E”]
        }
        • Example Response:
        {
        ”status”: ”success”
        }"""
    url = api_base_url+"/fox/send-message"
    data = {
        "teamId": team_id,
        "messages": messages,
        "message_entities": message_entities
    }
    res = requests.post(url, json=data)
    if res.ok:
        print(res.text)
    else:
        print("An Error in Send msg", res.status_code)

   
def end_fox(team_id):
    '''
    Use this function to call the api end point of ending the fox game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    2. Calling it without sending all the real messages will also affect your scoring fucntion
      (Like failing to submit the entire messageres.status_code within the timelimit of the game).
    '''
    """
     Endpoint: /fox/end-game
    • Method: POST
    • Parameters:
    – teamId (string): The ID of the team participating in the game.
    • Description: This API is used to end the game for the Fox. It concludes the game
    and provides the final score.
    • Response:
    – return text (string): Text indicating the score and whether it’s a new high score.
    Example Request:
    {
    ”teamId”: ”team123”
    }
    • Example Response:
    ”Game ended successfully with a score of 10. New Highscore reached!”
    """
    url = api_base_url+"/fox/end-game"
    data = {
        "teamId": team_id
    }
    res = requests.post(url, json=data)
    if res.ok:
        print(res.text)
    else:
        print("An Error occurred! status code:", res.status_code)

def submit_fox_attempt(team_id):
    '''
     Call this function to start playing as a fox. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as a Fox In phase1.
     In this function you should:
        1. Initialize the game as fox
        2. Solve riddles 
        3. Make your own Strategy of sending the messages in the 3 channels
        4. Make your own Strategy of splitting the message into chunks
        5. Send the messages 
        6. End the Game
    Note that:
        1. You HAVE to start and end the game on your own. The time between the starting and ending the game is taken into the scoring function
        2. You can send in the 3 channels any combination of F(Fake),R(Real),E(Empty) under the conditions that
            2.a. At most one real message is sent
            2.b. You cannot send 3 E(Empty) messages, there should be atleast R(Real)/F(Fake)
        3. Refer To the documentation to know more about the API handling 
    '''
    global api_base_url
    global model

    real_message, carrier_image = init_fox(team_id)

    riddle_id = "problem_solving_easy"
    # get the riddle
    test_case = get_riddle(team_id, riddle_id)
    # solve the riddle
    inc, budget, status = solve_riddle(team_id, riddle_solvers[riddle_id](test_case))
    if inc == 1: print("the soluion of the medium problem solving riddle is correct")
    else: print("the soluion of the medium problem solving riddle is wrong")

    riddle_id = "problem_solving_medium"
    test_case = get_riddle(team_id, riddle_id)
    inc, budget, status = solve_riddle(team_id, riddle_solvers[riddle_id](test_case))
    if inc == 2: print("the soluion of the medium problem solving riddle is correct")
    else: print("the soluion of the medium problem solving riddle is wrong")

    riddle_id = "problem_solving_hard"
    test_case = get_riddle(team_id, riddle_id)
    inc, budget, status = solve_riddle(team_id, riddle_solvers[riddle_id](test_case))
    if inc == 3: print("the soluion of the medium problem solving riddle is correct")
    else: print("the soluion of the medium problem solving riddle is wrong")

    riddle_id = "ml_easy"
    test_case = get_riddle(team_id, riddle_id)

    inc, budget, status = solve_riddle(team_id, riddle_solvers[riddle_id](test_case, model))
    if inc == 1: print("the soluion of the medium problem solving riddle is correct")
    else: print("the soluion of the medium problem solving riddle is wrong")

    encoded_messages, message_entities = generate_message_array(real_message, carrier_image)
    send_message(team_id, encoded_messages, message_entities)




    
    end_fox(team_id)

#submit_fox_attempt(team_id)


