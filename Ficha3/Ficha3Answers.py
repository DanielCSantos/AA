import pickle



def main():
    answers()


def answers():
    answer = pickle.load(open("A32078_Ficha3_Respostas.p", "rb"))

    # A)

    answer["Q001"][0][0] = False
    answer["Q001"][0][1] = True
    answer["Q001"][0][2] = False
    answer["Q001"][0][3] = False

    # B)

    answer["Q001"][1][0] = False
    answer["Q001"][1][1] = False
    answer["Q001"][1][2] = True
    answer["Q001"][1][3] = False

    # C)

    answer["Q001"][2][0] = False
    answer["Q001"][2][1] = False
    answer["Q001"][2][2] = True
    answer["Q001"][2][3] = False

    # D)

    answer["Q001"][3][0] = False
    answer["Q001"][3][1] = False
    answer["Q001"][3][2] = True
    answer["Q001"][3][3] = False

    # E)

    answer["Q001"][4][0] = True
    answer["Q001"][4][1] = False
    answer["Q001"][4][2] = False
    answer["Q001"][4][3] = False

    # F)

    answer["Q001"][5][0] = False
    answer["Q001"][5][1] = True
    answer["Q001"][5][2] = False
    answer["Q001"][5][3] = False

    # G)

    answer["Q001"][6][0] = True
    answer["Q001"][6][1] = False
    answer["Q001"][6][2] = False
    answer["Q001"][6][3] = False

    # H)

    answer["Q001"][7][0] = False
    answer["Q001"][7][1] = False
    answer["Q001"][7][2] = True
    answer["Q001"][7][3] = False

    # I)

    answer["Q001"][8][0] = False
    answer["Q001"][8][1] = False
    answer["Q001"][8][2] = False
    answer["Q001"][8][3] = True

    # J)

    answer["Q001"][9][0] = False
    answer["Q001"][9][1] = False
    answer["Q001"][9][2] = False
    answer["Q001"][9][3] = False

    pickle.dump(answer, open("A32078_Ficha3_Respostas.p", "wb"))


if __name__ == '__main__':
    main()




