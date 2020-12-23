# Dataset organization

## Frames Organization

The images frame extracted from the takes are logically allocated in the folders as follow below. However, to make the collaboration easy, we split these folders and organize them into directories with the name of each person that need do the annotation.

take1 -> First take
take2 -> Carro na pista
take2 -> Pessoa e carro na pista
take3 -> Carro em zig-zag, pessoa e carro na pista.
take4 -> Carro em zig-zag, ônibus na pista
take5 -> Carro em zig-zag, carro

## Procedure

Each person need to have the python pip installed in your machine, and then install the tool labelme. In Ubuntu the command to install is:

```bash
sudo -H pip3 install labelme
```
or
```bash
sudo -H pip install labelme
```


## Labels

All the annotators should use the same labels for the annotation. The labels are:

* road -> for the path where the car can traffic;
* person -> for people in the road
* car -> the cars on the road;
* bus -> for the bus on the road; and
* vegetation -> for all the part that aren't road or other things.

Take attention: Each label for countable things (like car, bus, person), in the same frame, need be assigned with an "-" ID for instance (the currency occurrence the frame), e.g. car-0, car-1, .... person-0, person-1, ... bus-0. For the other staffs, i.e. uncountable things or things that we don't need count (like road and vegetation), we use only the label. 





