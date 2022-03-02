import ipywidgets as wg
from IPython.display import Image

def mock_experiment(Slides):
    if Slides == 1:
        return Image(filename='../images/Slide1.PNG')
    elif Slides == 2:
        return Image(filename='../images/Slide2.PNG')
    elif Slides == 3:
        return Image(filename='../images/Slide3.PNG')
    elif Slides == 4:
        return Image(filename='../images/Slide4.PNG')
    elif Slides == 5:
        return Image(filename='../images/Slide5.PNG')
    else:
        return Image(filename='../images/Slide6.PNG')

def error_types(Slides):
    if Slides == 1:
        return Image(filename='../images/Slide7.PNG')
    elif Slides == 2:
        return Image(filename='../images/Slide8.PNG')
    elif Slides == 3:
        return Image(filename='../images/Slide9.PNG')
    elif Slides == 4:
        return Image(filename='../images/Slide10.PNG')
    elif Slides == 5:
        return Image(filename='../images/Slide11.PNG')
    else:
        return Image(filename='../images/Slide12.PNG')
    
def polymer_model(Slides):
    if Slides == 1:
        return Image(filename='../images/Slide13.PNG')
    elif Slides == 2:
        return Image(filename='../images/Slide14.PNG')
    elif Slides == 3:
        return Image(filename='../images/Slide15.PNG')
    elif Slides == 4:
        return Image(filename='../images/Slide16.PNG')
    elif Slides == 5:
        return Image(filename='../images/Slide17.PNG')
    else:
        return Image(filename='../images/Slide18.PNG')

def polymer_skip(Slides):
    if Slides == 1:
        return Image(filename='../images/Slide19.PNG')
    elif Slides == 2:
        return Image(filename='../images/Slide20.PNG')
    elif Slides == 3:
        return Image(filename='../images/Slide21.PNG')
    else:
        return Image(filename='../images/Slide22.PNG')
    
def find_polymer(Slides):
    if Slides == 1:
        return Image(filename='../images/Slide23.PNG')
    elif Slides == 2:
        return Image(filename='../images/Slide24.PNG')
    elif Slides == 3:
        return Image(filename='../images/Slide25.PNG')
    else:
        return Image(filename='../images/Slide26.PNG')
    
    