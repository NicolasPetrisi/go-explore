from goexplore_py.cell_representations import CellRepresentationBase
from typing import Set, Dict


class CellMapping:
    """Works as a two way dictionary where it is a normal dictionary, 
    but each value backtracks to all keys pointing to it. It is used to map from one cell to another and create so-called Hampu Cells.
    To create the illusion of larger cells in Go-Explore, this class re-links one cell to another such when
    a cell is referenced, the actual cell it is mapped to can be reached from here.

    Raises:
        RuntimeError: If a value is trying to be set to a key not existing in the dictionary already. New keys must always be added with the 'add_cell' method.
    """
    # The core dictionary of the class.
    __cell_mapping: Dict[CellRepresentationBase, CellRepresentationBase] = dict()

    # The reverse mapping from the values to each key pointing to them.
    __reverse_cell_mapping: Dict[CellRepresentationBase, Set[CellRepresentationBase]] = dict()

    def keys(self):
        """Get all keys from the core dictionary.

        Returns:
            Dictkeys[CellRepresentationBase]: The keys of the core dictionary.
        """
        return self.__cell_mapping.keys()


    def values(self):
        """Returns the values from the core dictionary.

        Returns:
            Dictvalues[CellRepresentationBase]: The values of the core dictionary.
        """
        return self.__cell_mapping.values()

        
    def get_cell_size(self, cell):
        """Get the cell size of the provided cell, how many cells does the given cell represent.
        If the cell does not exist in the archive, it is assumed to be 1.

        Args:
            cell (CellRepresentationBase): The cell to get the size of.

        Returns:
            int: The cell size.
        """
        if cell in self.__reverse_cell_mapping:
            return len(self.__reverse_cell_mapping[cell])
        else:
            return 1

    def get_mapping(self):
        """Returns a copy of the core dictionary.

        Returns:
            Dict[CellRepresentationBase, CellRepresentationBase]: The core dictionary.
        """
        return dict(self.__cell_mapping)

    def __getitem__(self, key):
        """Allows the class to be indexed using [] to get items like a normal dictionary.

        Args:
            key (CellRepresentationBase): The cell which you want the value of.

        Returns:
            CellRepresentationBase: The cell which the given cell maps to.
        """
        if key in self.__cell_mapping:
            return self.__cell_mapping[key]
        else:
            return key


    def add_cell(self, cell):
        """Adds a cell to the cell mapping. Any new cell entering the cell mapping must be added through here.
        Each newly added cell will be linked to itself.

        Args:
            cell (CellRepresentationBase): The cell to add.
        """
        if cell in self.__cell_mapping:
            return

        self.__cell_mapping[cell] = cell
        self.__reverse_cell_mapping[cell] = set([cell])


    def load_cell_mapping(self, cell_mapp: Dict[CellRepresentationBase, CellRepresentationBase]):
        """Load an pre-existing cell mapping 

        Args:
            cell_mapp (Dict[CellRepresentationBase, CellRepresentationBase]): The core dictionary to load and base itself on.
        """
        self.__cell_mapping = dict(cell_mapp)
        for k, v in self.__cell_mapping.items():
            if v not in self.__reverse_cell_mapping:
                self.__reverse_cell_mapping[v] = set([k])
            else:
                self.__reverse_cell_mapping[v].add(k)


    def __setitem__(self, key_cell, value_cell):
        """Allows the class to use indexing to set items: A[x] = y

        Args:
            key_cell (CellRepresentationBase): The cell to use as key.
            value_cell (CellRepresentationBase): The cell to use as value.

        Raises:
            RuntimeError: If the item does not already exist in the cell mapping. Always add cells using 'add_cell' before calling this with the cell as key!
        """
        if key_cell not in self.__cell_mapping:
            raise RuntimeError("key_cell does not already exist in __cell_mapping. Add the key using 'add_cell' first!")

        if self.__cell_mapping[key_cell] == value_cell:
            return

        try:
            children = self.__reverse_cell_mapping[key_cell]
        except:
            print("CRASHING")
            print("This is cell_mapping:")
            for k, v in self.__cell_mapping:
                print(k, ":", v)
            print("\n\n")
            print("This is reverse_mapping:")
            for k, v in self.__reverse_cell_mapping:
                print(k, ":", v)
            raise RuntimeError("WARNING")



        self.__reverse_cell_mapping[key_cell] = set()
        for child in children:
            self.__cell_mapping[child] = value_cell
            self.__reverse_cell_mapping[value_cell].add(child)


    def __contains__(self, item):
        """Allows the class to use the 'in' keyword.

        Args:
            item (CellRepresentationBase): The item to check if is here.

        Returns:
            bool: If the given item exists or not in the core dictionary.
        """
        return item in self.__cell_mapping

    def __repr__(self):
        return f'{self.__cell_mapping}'

