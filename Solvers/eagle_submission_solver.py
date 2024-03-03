import numpy as np
from LSBSteg import decode
import tensorflow as tf
import requests

api_base_url = 'http://16.171.171.147:5000'
team_id= "bVUrA0A"
model_path = 'cnn_dell.h5'
loaded_model = tf.keras.models.load_model(model_path)


def init_eagle(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''
    global loaded_model
    loaded_model = tf.keras.models.load_model(model_path)

    '''
                    3.1Start Game
                    • Endpoint: /eagle/start
                    • Method: POST
                    • Parameters:
                    – teamId (string): The ID of the team participating in the game.
                    • Description: This API is used to start the game for a specific team. It initializes the
                    game and returns the first set of footprints.
                    • Response:
                    – footprint : An array of three footprints represented as NumPy spectrograms.
                    • Example Request:
                    {
                    ”teamId”: ”team123”
                    }
                    • Example Response:
                    {
                    ”footprint”: {”1”: spectrogram1, ”2”:spectrogram2, ”3”:spectrogram3 }
                    }
        '''
    global api_base_url
    url = api_base_url+"/eagle/start"
    data = {
        "teamId": team_id
    }
    res = requests.post(url, json=data)
    if res.status_code == 200:
        res = res.json()
        footprint = res["footprint"]
        print(f"init_eagle {res}")
        return footprint
    else:
        print("An Error occurred in init_eagle! status code:", res.status_code)
        return None




def preprocess_input_data(input_data):
    """
    Preprocess input data for inference.

    Parameters:
    - input_data: Raw input data

    Returns:
    - processed_data: Preprocessed input data
    """

    # Extract and preprocess features as needed
    processed_data = np.array([np.array(input_data[str(i)]) for i in range(1, 4)])
    processed_data = np.clip(processed_data, a_min=None, a_max=1e2)
    processed_data = processed_data / 255.0
    
    return processed_data

def infer(input_data):
    global loaded_model
    # Preprocess input data
    pro_data = preprocess_input_data(input_data)
    input_data_reshaped = pro_data.reshape((pro_data.shape[0],) + pro_data.shape[1:] + (1,))

    # Perform inference
    predictions_prob = loaded_model.predict(input_data_reshaped)

    return predictions_prob

def select_channel(footprint):
    '''
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.        
    '''
    probabilities = infer(footprint)
    threshold = 0.5
    if max(probabilities) > threshold:
        return list(probabilities).index(max(probabilities)) + 1
    else:
        return -1


  
def skip_msg(team_id:str):
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    '''
    • Endpoint: /eagle/skip-message
    • Method: POST
    • Parameters:
    – teamId (string): The ID of the team participating in the game.
     Description: This API is used to skip through all messages in the current chunk and
    move on to the next set. Used in case all footprints were detected to be fake/empty.
    • Response:
    – nextFootprint : The next chunk’s footprints - an array of three footprints represented as NumPy spectrograms. Each spectrogram is received as a list that
    should later be converted to a NumPy array using np.array(). If the end of the
    message is reached, you will be notified that no more footprints exist and you
    should then end game.
    • Example Request:
    {
    ”teamId”: ”team123”
    }
    • Example Response:
    If there exsist more footprints:
    {
    ”nextFootprint”:{”1”: spectrogram1, ”2”:spectrogram2, ”3”:spectrogram3 }
    }
    If no more footprint exist:
    ”End of message reached”
    '''
    global api_base_url
    url = api_base_url + "/eagle/skip-message"
    data = {"teamId": team_id}

    try:
        # Make a POST request to skip the message
        res = requests.post(url, json=data)

        if res.status_code == 200:
            res_json = res.json()
            print(f"skip_msg {res}")


            if "nextFootprint" in res_json:
                next_footprint_data = res_json["nextFootprint"]
                return next_footprint_data
            else:
                print("End of message reached")
                return None

        else:
            print("An error occurred in skip_msg! Status code:", res.status_code)
            return None

    except requests.exceptions.RequestException as err:
        print("Request Error:", err)
        return None
  


def request_msg(team_id:str, channel_id:int):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''
    """
    • Endpoint: /eagle/request-message
    • Method: POST
    • Parameters:
    – teamId (string): The ID of the team participating in the game.
    – channelId (integer): The channel number (1, 2, or 3) from which to request the
    message.
    • Description: This API is used to request a message from a specific channel in the
    current set of footprints. This must be followed with either /skip-message or /submitmessage.
    • Response:
    – encodedMsg (numpy array): The requested message from the specified channel,
    in the form of a numpy array.
    • Example Request:
    {
    ”teamId”: ”team123”
    ”channelId”: 2
    }
    • Example Response:
    {
    ”encodedMsg”: [[0.2 0.4 0.6] [0.3 0.5 0.7], [0.1 0.8 0.9]]
    }
    """
    global api_base_url
    url = api_base_url+"/eagle/request-message"
    data = {
        "teamId": team_id,
        "channelId": channel_id
    }
    res = requests.post(url, json=data)
    if res.status_code == 200:
        res = res.json()
        encodedMsg = res["encodedMsg"]

        print(f"request_msg {res}")

        return encodedMsg
    else:
        print("An Error in request_msg! status code:", res.status_code)
        return None



def submit_msg(team_id:str, decoded_msg:str):
    '''
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message  
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    """
    • Endpoint: /eagle/submit-message
    • Method: POST
    • Parameters:
    – teamId (string): The ID of the team participating in the game.
    – decodedMsg (string): The decoded message.
    • Description: This API is used to submit the decoded message - the result of decoding
    the message previously requested.
    • Response:
    – nextFootprint : The next chunk’s footprints - an array of three footprints represented as NumPy spectrograms. Each spectrogram is received as a list that
    should later be converted to a NumPy array using np.array(). If the end of the
    message is reached, you will be notified that no more footprints exist and you
    should then end game.
    • Example Request:
    {
    ”teamId”: ”team123”
    ”decodedMsg”: ”Decoded message”
    }
    • Example Request:
    {
    ”teamId”: ”team123”
    }
    • Example Response:
    If there exsist more footprints:
    {
    ”nextFootprint”:{”1”: spectrogram1, ”2”:spectrogram2, ”3”:spectrogram3 } }
    If no more footprint exist:
    ”End of message reached”
    """
    global api_base_url
    url = api_base_url+"/eagle/submit-message"
    data = {
        "teamId": team_id,
        "decodedMsg": decoded_msg
    }
    res = requests.post(url, json=data)
    if res.status_code == 200:
        res = res.json()
        print(f"submit_msg {res}")
        if "nextFootprint" in res:
            next_footprint = res["nextFootprint"]
            return next_footprint
        else:
            print("End of message reached")
            return None
    else:
        print("An Error in submit_msg! status code:", res.status_code)
        return None





  
def end_eagle(team_id:str):
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''
    """
    Endpoint: /eagle/end-game
    • Method: POST
    • Parameters:
    – teamId (string): The ID of the team participating in the game.
    • Description: This API is used to end the game for the eagle. It concludes the game
    and provides the final score.
    • Response:
    – return text (string): Text indicating the score and whether it’s a new high
    score.
    • Example Request:
    {
    ”teamId”: ”team123”
    }
    • Example Response:
    ”Game ended successfully with a score of 10. New Highscore reached!”
    """
    global api_base_url
    url = api_base_url+"/eagle/end-game"
    data = {
        "teamId": team_id}
    res = requests.post(url, json=data)
    if res.status_code == 200:
        print(f"end {res}")

    else:
        print("An Error in end_eagle! status code:", res.status_code)
        return None

def submit_eagle_attempt(team_id:str):
    '''
     Call this function to start playing as an eagle. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as an Eagle In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve the footprints to know which channel to listen on if any.
        3. Select a channel to hear on OR send skip request.
        4. Submit your answer in case you listened on any channel
        5. End the Game
    '''
    footprint = init_eagle(team_id)
    while footprint:
        channel_id = select_channel(footprint)
        if channel_id == -1:
            footprint = skip_msg(team_id)
        else:
            encoded_msg = request_msg(team_id, channel_id)
            decoded_msg = decode(encoded_msg)
            footprint = submit_msg(team_id, decoded_msg)

    end_eagle(team_id)

#submit_eagle_attempt(team_id)
    
url ="http://13.53.169.72:5000/attempts/professional"
data = {
        "teamId": team_id
        }
res = requests.post(url, json=data)
print(res.json())
