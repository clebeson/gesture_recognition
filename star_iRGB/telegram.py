import time
import requests


def telegram_bot_sendtext(bot_message):
    print("sending ...")
    bot_token = '*********'
    bot_chatID = '********'
    
    # send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message

    send_text = "https://api.telegram.org/bot{}/sendMessage?chat_id={}&text={}".format(bot_token,bot_chatID,bot_message)
    response = requests.get(send_text)

    return response.json()


