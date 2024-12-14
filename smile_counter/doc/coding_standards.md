# SMILE COUNTER CODING STANDARDS

**Follow [PEP-8](https://peps.python.org/pep-0008/) coding standards for assumptions not defined in this file.**

Try to utilise [clean code](https://gist.github.com/wojteklu/73c6914cc446146b8b533c0988cf8d29) and [SOLID](https://en.wikipedia.org/wiki/SOLID) principles when possible.


# General conventions
1. **Use double quotes for strings.**

    ```python 
    # correct:
    my_str = "test"

    # wrong:
    my_str = 'test'
    ```

2. **Use F-string formatting instead of old C-style "%-formatting" (consistency with [PEP-3101](https://peps.python.org/pep-3101/#abstracthttps://peps.python.org/pep-0008/)  standard).**

    ```python 
    # correct:
    my_formatted_str = f"This is {'f'.capitalize()}-string"

    # wrong:
    my_formatted_str = "This is %s string" % ("C-style")
    ```

3. **Use triple double quotes for docstrings (consistency with [PEP-257](https://peps.python.org/pep-0257/#what-is-a-docstring) standard).**

    ```python 
    # correct:
    def func:
        """
        This is docstring
        """

    # wrong:
    def func:
        '''
        This is docstring
        '''
    ```

4. **Use max line length of *120* characters.**

5. **Do not use a backslash to break the import line, use brackets to grup imported classes (consistency with [PEP-328](https://peps.python.org/pep-0328/#rationale-for-parentheses) standard).**

    ```python 
    # correct:
    def func:
    from Tkinter import (Tk, Frame, Button, Entry, Canvas, Text,
        LEFT, DISABLED, NORMAL, RIDGE, END)

    # wrong:
    from Tkinter import Tk, Frame, Button, Entry, Canvas, Text, \
        LEFT, DISABLED, NORMAL, RIDGE, END
    ```


6. **Use type annotations to supports type hints by IDEs (consistency with [PEP-484](https://peps.python.org/pep-0484/#abstract)).**

    ```python 
    # correct:
    def func(a: int) -> list[int]:
    
    def start_the_app(app_ready: bool) -> None:


    # wrong:
    def func(a):

    def start_the_app(app_ready):
    ```

7. **Use uppercase letters for const variables to be consistent with [PEP-8](https://peps.python.org/pep-0008/).**

    ```python 
    # correct:
    CONST_VARIABLE = 0
    
    # wrong:
    const_variable = 0
    ```

8. **Use *snake_case* convention for variables/methods to be consistent with [PEP-8](https://peps.python.org/pep-0008/).**

    ```python 
    # correct:
    this_is_an_example = is_true_function()
    
    # wrong:
    thisIsAnExample = IsTrueFunction()
    ```

9. **Use underscore(_) beofre the name of private/protected attribute/function of the class/module to mark it as non-public. this practice is not only a convention, in case of wild card (*) imports function/variables marked as non-public will not be imported. For access level control follow the [principle of least astonishment](https://en.wikipedia.org/wiki/Principle_of_least_astonishment).** 

    **That means you should not use use getters and setters that should be accessible for both read and write without any restrictions (mark them as public). If you wish to add some restrictions then you should use proper decorator (@property - part of language syntax) to implement getters/setters.**

    **DO NOT use ```get_variable_name``` and ```set_variable_name``` as it may be recommended for other programming languages, because it may lead to some confusion caused by internal ```getattr``` and ```setattr``` methods existence.**


    ```python 
    class Example:
        def __init__(self, a: int, b: int, c: int, d: int, e :int) -> None:
            self._a = a # private/protected attribute, not-exposed out of the class scope
            self._b = b # private/protected attribute for read-only
            self._c = c # private/protected attribute for write-only
            self.d = d # public attribute exposed for both read and write without any restrictions
            self._e # public attribute exposed for both read and write with some restrictions

    @property
    def b(self) -> int:
        # read-only access - getter method
        return self._b

    @c.setter
    def c(self, value: int) -> None:
        # write-only access - setter method
        self._c = value

    @property
    def e(self) -> int:
        # read access - getter method
        return self._e

    @e.setter
    def e(self, value: int) -> None:
        # write acces with some restrictions - setter method
        # add some restrictions
        # 1. verify the value type
        if not isinstance(value, int):
            return TypeError(
                "'e' attribute of the {cls_name} class must be an integer,"
                " passing other type of argument prevents the class from working properly,"
                " new value: {new_value} was not set for the attribute, will continue with"
                " previous value: {old_value}".format(cls_name=repr(self.__class__.__name), 
                new_value=repr(value), old_value=repr(self._e))
            )
        # 2. verify the value range
        if not (0 <= value <= 100):
            raise ValueError(
                "'e' attribute of the {cls_name} class represents percentage so it must fit into,"
                " proper range between 0 and 100 inclusive, new value: {new_value} was not set"
                " for the attribute, will continue with previous value: {old_value}".format(cls_name=repr(self.__class__.__name), 
                new_value=repr(value), old_value=repr(self._e))
            )
        # all checks passed, set the attribute
        self._e = value

    def _priv_method(self) -> None:
        print("This is private method, not exposed externally")
    
    def pub_method(self) -> None:
        print("This is public method exposed externally")



    x = Example(0,1,2,3,4)

    # correct:

    # - b: read only attr
    y = x.b
    # - c : write only attr
    x.c = y
    # - d : write and read attr
    x.d = y
    y = x.d
    # - e: write and read the restriction attr
    x.e = 99
    y = x.e
    # - public method
    x.public_method()


    # wrong:
    
    # - a: private attr
    x._a = 0
    x.a = 0 # takes no effect
    y = x._a
    y = x.a
    # will cause
    # >>> Traceback (most recent call last):
    # >>> File "<stdin>", line 1, in <module>
    # >>> AttributeError: 'Example' object has no attribute 'a'

    # - b: read only str
    x._b = y
    x.b = y
    y = x._b
    # will cause
    # >>> Traceback (most recent call last):
    # >>> File "<stdin>", line 1, in <module>
    # >>> AttributeError: can't set attribute

    # - c: write only attr
    x._c = 0
    y = x._c
    y = x.c
    # will cause
    # >>> Traceback (most recent call last):
    # >>> File "<stdin>", line 1, in <module>
    # >>> AttributeError: unreadable attribute

    # - e: write and read with restrictions attr
    x._e = 0
    x.e = -1
    x.e = 101
    x.e = ''
    y = x._e
    # will cause one of the 'e' method exceptions
    
    # - methods
    x._priv_method()
    # interpreter will show you only public methods/attributes:
    # >>> x.
    # x.b       x.c     x.d     x.e     x.pub_method()

    ```

10. **Do not use [magic numbers](https://en.wikipedia.org/wiki/Magic_number_(programming)) or [magic strings](https://en.wikipedia.org/wiki/Magic_number_(programming)). Instead use named constants (also called explanatory variables) or enums (enumeration class).**

    ```python 
    # correct:
    OWNER_NAME = "Adam"
    MAX_NAME_LENGTH = 40

    if len(self._name) > MAX_NAME_LENGTH:
        raise ValueError(
            "{name} has exceeded allowed length limit."
            "Name must be shorter than {max_len} characters."
            "Contact {owner} if you need assistance.".format(name=repr(self._name), 
            max_len=repr(MAX_NAME_LENGTH), owner=repr(OWNER_NAME))
        )
    

    # wrong:
    if len(self._name) > 40:
        raise ValueError(
            "{name} has exceeded allowed length limit."
            "Name must be shorter than 40 characters."
            "Contact Adam if you need assistance.".format(name=repr(self._name))
        )
    ```

11. **Use underscore (_) for elements that should be ignored in a loop during processing.**

    ```python 
    # correct:
    for _, value in { "key1": value1, "key2": value2, "key3": value3}.items():
        print(value)
    
    # wrong:
        for key, value in { "key1": value1, "key2": value2, "key3": value3}.items():
        # 'key' variable is not used!
        print(value)
    ```

12. **Name properly items taken from collection during the iteration process, ```i``` variable name stands for ```index```, as the name suggests this shortcut is widely used during iteration over array items, which is fine however it should not be applied for each iteam in each loop. The name should always reflect the exact content of it.**

    ```python 
    # correct:
    for i in range(len(example)):
        # - "i" is used as index, so the name of the variable reflects
        # the content of the variable properly
        print(f"example[{i}] = {example[i]}")
    
    # correct:
    for letter_sign in example:
        # - the name of the variable reflects correctly the content of the item
        print(f"{letter_sign}")
    
    # wrong:
    for i in example:
        # - "i" is not used as index, so the name of the variable doesn't reflects
        # the content of the variable properly
        print(f"{[i]}")

    ```

13. **Avoid usage of public methods in other public methods of the same module. These are exposed for ```external``` usage.**

    ```python 
    # correct:
    def _priv_shared_logic():
        print("some logic")
    def public_method_0():
        _priv_shared_logic()
    def public_method_1():
        _priv_shared_logic()
        _do_other_stuff()

    
    # wrong:
    def public_method_0():
        print("some logic")
    def public_method_1():
        public_method0()
        _do_other_stuff()

    ```
    **It makes code messy ([Spaghetti code](https://en.wikipedia.org/wiki/Spaghetti_code)), follow the rule to call public methods only from ```external``` modules**
    **Otherwise unstructured code may bring problems, not easy to notice at first glance, for example:**

    ```python 
    # unsuccessful attempt to implement logic which causes circular dependency 0 -> 2 -> 1 -> 0
    def public_method_0():
        _do_step_0()
        public_method_2()
    def public_method_1():
        _do_step_1()
        public_method_0()
    def public_method_2():
        _do_step_2()
        public_method_1()
        external.module.public_external_method()
    ```

    **If we will keep the rule to use only private methods at the same level, code will be cleaner, and less error-prone:**
    ```python 
    # unsuccessful attempt to implement logic which causes circular dependency 0 -> 2 -> 1 -> 0
    def public_method_0():
        _do_step_0()
        _do_step_2()
        _do_step_1()
    def public_method_1():
        _do_step_1()
        _do_step_0()
        _do_step_2()
    def public_method_2():
        _do_step_2()
        _do_step_1()
        _do_step_0()
        external.module.public_external_method()
    ```

13. **WIP**


# Indentation


# Docstrings


# Logging


# Imports
