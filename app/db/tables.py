from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Date, Integer, String
from sqlalchemy.ext.declarative import declarative_base

class DataBase:
    engine = create_engine('sqlite:///database.db', echo=True)
    Base = declarative_base()


    class Crypto(Base):
        __tablename__ = "crypto"
        id = Column(Integer, primary_key=True)
        name = Column(String)  
        ticker = Column(String)  

        def __init__(self, name):
            self.name = name    
    
    def init_app(self):
        self.Base.metadata.create_all(self.engine)