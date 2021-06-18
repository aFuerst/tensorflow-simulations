from nano import nano_init
class Main:
    def __init__(self):
        # self.master = execute.Execute()
        return

    def launch(self):
        # Start executing MD simulation steps here
        # status = self.master.run_simulation()
        nano_init.start()

        # Start training the surrogate with generated data
        # if status:
        #     self.master.run_surrogate()
        # else:
            #TODO:Log that MD simulations sis not finish successfully
            # pass
        return


if __name__ == "__main__":
    m = Main()
    m.launch()