'''
Multiprocessing messaging system
'''

class MseMessage:
    def __init__(self, actorId=None, name=None):
        self.actorId = actorId
        self.__name__ = name


    def __str__(self):
        if self.__name__ is None:
            return("Multiprocessing abstract message")
        else:
            return("Multiprocessing abstract message: {}".format(self.__name__))


    def __repr__(self):
        return(self.__str__())



class PoisonMessage(MseMessage):
    def __init__(self, **kwargs):
        super(PoisonMessage, self).__init__(**kwargs)


    def __str__(self):
        if self.__name__ is None:
            return("Multiprocessing poison message")
        else:
            return("Multiprocessing poison message: {}".format(self.__name__))



class InfoMessage(MseMessage):
    def __init__(self, text, **kwargs):
        super(InfoMessage, self).__init__(**kwargs)
        self.text = text


    def __str__(self):
        if self.__name__ is None:
            return("Multiprocessing info message")
        else:
            return("Multiprocessing info message: {}".format(self.__name__))



class DataMessage(MseMessage):
    def __init__(self, data, **kwargs):
        super(DataMessage, self).__init__(**kwargs)
        self.data = data


    def __str__(self):
        if self.__name__ is None:
            return("Multiprocessing data message")
        else:
            return("Multiprocessing data message: {}".format(self.__name__))




