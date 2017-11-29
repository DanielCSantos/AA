import pickle



def main():
    answers()


def answers():
    answer = pickle.load(open("A32078_Ficha2_Respostas.p", "rb"))

    answer["Q001"][0] = False
    answer["Q001"][1] = False
    answer["Q001"][2] = False
    answer["Q001"][3] = False

    # A)
    answer["Q002"][0] = False
    # B)
    answer["Q002"][1] = False
    # C)
    answer["Q002"][2] = False
    # D)
    answer["Q002"][3] = False
    # E)
    answer["Q002"][4] = False
    # F)
    answer["Q002"][5] = True

    # A i)
    answer["Q003_A"][0] = False
    # A ii)
    answer["Q003_A"][1] = True
    # A iii)
    answer["Q003_A"][2] = False
    # A iV)
    answer["Q003_A"][3] = True

    # B i)
    answer["Q003_B"][0] = False

    # B ii)
    answer["Q003_B"][1] = True

    # B iii)
    answer["Q003_B"][2] = False

    # B iV)
    answer["Q003_B"][3] = False

    # A
    answer["Q004"][0] = False
    # B
    answer["Q004"][1] = False
    # C
    answer["Q004"][2] = True
    # D
    answer["Q004"][3] = False

    pickle.dump(answer, open("A32078_Ficha2_Respostas.p", "wb"))

    print("oi")


if __name__ == '__main__':
    main()