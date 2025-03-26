import random
player_score = 0
computer_score = 0
def game_random():
    global player_score, computer_score
    won = False
    draw = False
    choices = ["K","N","P"]
    randomised = random.choice(choices)
    player_choice = input("Wybierz K dla Kamień, P dla Papier, lub N dla Nożyce: ")
    if player_choice not in choices:
        print("Błędny wybór")
        return None
        
    if player_choice == "K" and randomised == "N":
            won = True
            draw = False
    elif player_choice == "K" and randomised == "P":
            won = False
            draw = True
    elif player_choice == "K" and randomised == "K":
            won = False
            draw = True
    elif player_choice == "P" and randomised == "N":
            won = False
            draw = False
    elif player_choice == "P" and randomised == "P":
            won = False
            draw = True
    elif player_choice == "P" and randomised == "K":
            won = True
            draw = False
    elif player_choice == "N" and randomised == "N":
            won = False
            draw = True
    elif player_choice == "N" and randomised == "P":
            won = True
            draw = False
    elif player_choice == "N" and randomised == "K":
            won = False
            draw = False
    if won == True and draw == False:
            player_score += 1
            print("wygrana")
    elif won == False and draw == True:
            print("remis")
    elif won == False and draw == False:
            computer_score += 1
            print("przegrana")

while True:
    game_random()
    print("Wynik: " + str(player_score) + " do: " + str(computer_score))
    print("grać dalej?")
    cont =  input("T dla tak, N dla nie: ")
    if cont != "T":
        print("Koniec gry")
        break