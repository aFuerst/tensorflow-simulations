import api.main.execute
class Main:
    def __init__(self):
        self.master = api.execute.Execute()
        return

    def launch(self):
        # Start executing MD simulation steps here
        status = self.master.run_simulation()

        # Start training the surrogate with generated data
        if status:
            self.master.run_surrogate()
        else:
            #TODO:Log that MD simulations sis not finish successfully
            pass
        return
