import random
player_score = 0
computer_score = 0
while True:
    choices = ["K","N","P"]
    randomised = random.choice(choices)
    def game_random(list):
        won = False
        draw = False
        player_choice = input("Wybierz K dla Kamień, P dla Papier, lub N dla Nożyce")
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
            player_score =+ 1
            print("wygrana")
            print("wynik gracza: " + str(player_score) + "wynik komutera: " + str(computer_score))
        if won == False and draw == True:
            print("remis")
            print("wynik gracza: " + str(player_score) + "wynik komutera: " + str(computer_score))
        if won == False and draw == False:
            computer_score =+ 1
            print("przegrana")
            print("wynik gracza: " + str(player_score) + "wynik komutera: " + str(computer_score))

    game_random(choices)
    print(game_random)
    print("grać dalej?")
    if input("T dla tak, N dla nie") == "T":
        game_random(choices)
        print(game_random)
    if input("T dla tak, N dla nie") == "N":
        print("OK")