from goexplore_py.cell_representations import CellRepresentationBase
from typing import Set, Dict


class CellMapping:
    __cell_mapping: Dict[CellRepresentationBase, CellRepresentationBase] = dict()
    __reverse_cell_mapping: Dict[CellRepresentationBase, Set[CellRepresentationBase]] = dict()

    def keys(self):
        return self.__cell_mapping.keys()


    def values(self):
        return self.__cell_mapping.values()

        
    def get_cell_size(self, cell):
        if cell in self.__reverse_cell_mapping:
            return len(self.__reverse_cell_mapping[cell])
        else:
            return 1


    def __getitem__(self, key):
        return self.__cell_mapping[key]


    def add_cell(self, cell):
        if cell in self.__cell_mapping:
            return

        self.__cell_mapping[cell] = cell
        self.__reverse_cell_mapping[cell] = set([cell])


    def __setitem__(self, key_cell, value_cell):

        if key_cell not in self.__cell_mapping:
            raise RuntimeError("key_cell does not already exist in __cell_mapping")

        if self.__cell_mapping[key_cell] == value_cell:
            return


        children = self.__reverse_cell_mapping[key_cell]
        for child in children:
            self.__cell_mapping[child] = value_cell
            self.__reverse_cell_mapping[value_cell].add(child)


        if key_cell in self.__reverse_cell_mapping and key_cell != value_cell:
            del self.__reverse_cell_mapping[key_cell]

        

    def __contains__(self, item):
        return item in self.__cell_mapping

    def __repr__(self):
        return f'{self.__cell_mapping}'

