'''
Generator to create Caffe Models.

It loads the model definition as yaml and generates a prototxt from it.
'''


import fire
import yaml
import os
import re
import datetime
from abc import ABCMeta, abstractmethod


class Block(metaclass=ABCMeta):
    '''Abstract class for the block that allows to load them.'''
    @abstractmethod
    def load(self, file, input=None, prefix=""):
        pass

    @abstractmethod
    def generate(self):
        pass


class YamlBlock(Block):
    '''Defines a specific block of the network.'''

    def __init__(self, params={}, name=None, desc="", debug=False):
        self.name = name
        self.blocks = []
        self.params = params
        self.description = desc
        self._debug = debug

    def _replace_eval(self, string, params, use_iter=True):
        # resolve all references
        re_ref = re.compile("::([A-Za-z0-9_\-\.]+)")
        offset = 0
        match = re_ref.search(string)
        while match is not None:
            skey = match.group(1)
            pos = match.span()

            # avoid special iter for later use case
            if skey == "ITER" and use_iter:
                offset = pos[1]
                match = re_ref.search(string, offset)
                continue

            if skey not in params:
                raise ValueError('Could not find referenced parameter ({}) in block ({})'.format(skey, self.name))

            # replace the reference

            string = string[:pos[0]] + str(params[skey]) + string[pos[1]:]

            # search next position
            match = re_ref.search(string, offset)

        return string

    def _iter_params(self, params, iter):
        '''Calculates the params for the current iteration.'''
        res = params.copy()
        for key in res:
            value = res[key].replace("::ITER", str(iter))
            if value.startswith("!!"):
                value = eval(value[2:])
            res[key] = value
        return res

    def _eval_item(self, item, params):
        '''Completely evaluates a single item.'''
        item = str(item)
        item = self._replace_eval(item, params, False)
        if item.startswith('!!'):
            item = eval(item[2:])
        return item

    def load(self, file, input=None, prefix=""):
        '''Parses a YAML file.

        If a model is provided it will extend the model.
        Otherwise it will create a new model from scratch.
        '''
        # safty: check that the file exists
        if not os.path.exists(file) or not os.path.isfile(file):
            raise IOError('Could not load the yaml file ({}). File not found!'.format(file))

        # Load the yaml file
        block_yaml = {}
        with open(file, 'r') as stream:
            try:
                block_yaml = yaml.load(stream)
            except yaml.YAMLError as exc:
                raise IOError('Could not parse the yaml file ({})'.format(exec))

        # load the name of the model
        if 'name' in block_yaml and self.name is None:
            self.name = block_yaml['name']
        # load the description of the block
        if 'description' in block_yaml and len(self.description) == 0:
            self.description = block_yaml['description']

        # load the parameters
        load_params = block_yaml['params'] if 'params' in block_yaml else {}
        # combine the parameters
        params = self.params
        for key in load_params:
            if key not in self.params:
                params[key] = load_params[key]

        # retrieve the path of the current file
        path_prefix = os.path.split(file)[0]

        # iterate through all yaml blocks
        for block in block_yaml['blocks']:
            # parse data
            next_file = os.path.join(path_prefix, block['file'])
            # update the repeat value to allow for eval expressions
            next_repeat = None
            if 'repeat' in block:
                next_repeat = int(self._eval_item(block['repeat'], params))

            # check if hidden and if so jump over it
            next_hidden = False
            if 'hide' in block:
                next_hidden = str(self._eval_item(block['hide'], params))
                next_hidden = next_hidden.lower() in ['1', 'true', 'yes']
            if next_hidden:
                continue

            # load the prefix
            next_prefix = block['prefix'] if 'prefix' in block else ""
            if len(prefix) > 0:
                if len(next_prefix) > 0:
                    next_prefix = prefix + "/" + next_prefix
                else:
                    next_prefix = prefix

            # load the parameters
            # NOTE: use parent params by default
            next_params = params.copy()
            if 'params' in block:
                for p in block['params']:
                    next_params[p] = block['params'][p]

            # check description
            next_desc = ""
            if 'description' in block:
                next_desc = block['description']

            # update the param values
            for key in next_params:
                next_params[key] = self._replace_eval(str(next_params[key]), params)

            # check the block type
            if block['type'] == 'proto':
                next_file += ".prototxt"
                next_block_lmd = lambda i: ProtoBlock(self._iter_params(next_params, i), block['name'], self._debug)
            elif block['type'] == 'yaml':
                next_file += ".yaml"
                next_block_lmd = lambda i: YamlBlock(self._iter_params(next_params, i), block['name'], next_desc, self._debug)
            elif block['type'] == 'predef':
                # TODO: implement predefined blocks with params here!
                print('NOT YET IMPLEMENTED (predef)!')
                pass
            else:
                print('ERROR: Unkown block type ({})'.format(block['type']))
                continue

            def update_input(block, input):
                # check the output mapping
                if 'output' in block:
                    for item in block['output']:
                        kin = input[1]
                        din = input[0]
                        key = item['in']
                        out = item['out']

                        # check if number and retrieve value
                        val = None
                        if type(key) is int or key.isdigit():
                            val = input[0][int(key)]
                        else:
                            val = kin[key]

                        # check if output references number and set value
                        if type(out) is int or out.isdigit():
                            if int(out) >= len(din):
                                din.append(val)
                            else:
                                din[int(out)] = val
                        else:
                            kin[out] = val
                        input = (din, kin)
                return input

            # load the data and append
            if next_repeat is not None:
                for i in range(next_repeat):
                    # generate the iteration prefix
                    iter_prefix = '{}_i{}'.format(next_prefix, i + 1) if len(next_prefix) > 0 else 'iter_{}'.format(i)
                    # add the current iteration
                    next_block = next_block_lmd(i + 1)
                    input = next_block.load(next_file, input, iter_prefix)
                    self.blocks.append(next_block)
                    input = update_input(block, input)
            else:
                # create the block
                next_block = next_block_lmd(1)
                input = next_block.load(next_file, input, next_prefix)
                self.blocks.append(next_block)
                input = update_input(block, input)

            if self._debug:
                print("layer: {:20}\n  list: {}\n  dict: {}".format(block['name'], input[0], input[1]))

        # retruns the last found output
        return input

    def generate(self):
        '''Generates a string from this block.'''
        prefix = ""
        if self.name is not None:
            prefix = "\n# --- {} ---\n# {}\n\n".format(self.name, self.description)
        return prefix + '\n'.join([b.generate() for b in self.blocks])


class ProtoBlock(Block):
    '''Defines a block that contains the '''

    def __init__(self, params={}, name=None, debug=False):
        self.params = params
        self.network = ""
        self.output = []
        self.output_dict = {}
        self.name = name
        self._debug = debug

        # define special matching chars
        #self.special = ['NUM', 'OUTPUT']

    def _replace_string(self, regex, string, fct, nextpos=False):
        '''Generator to replace the regex with the value from fct in a string.'''
        # create the pattern
        match = regex.search(string)

        # match the data
        while match is not None:
            # create the value
            value = fct(match)

            # replace
            pos = match.span()
            string = string[:pos[0]] + str(value) + string[pos[1]:]
            # output
            yield string

            # continue search
            match = regex.search(string, pos[0] + len(value) if nextpos else 0)

    def _replace_string_all(self, regex, string, fct, nextpos=False):
        '''Replaces all items of the regex in the given string.'''
        gen = self._replace_string(regex, string, fct, nextpos)
        output = string
        for net in gen:
            output = net
        return output

    def load(self, file, input=None, prefix=""):
        '''Parses the prototxt file and adds the stuff to the model.'''
        # safty: check that the file exists
        if not os.path.exists(file) or not os.path.isfile(file):
            raise IOError('Could not load the proto file ({}). File not found!'.format(file))

        # load proto as single text
        network = ""
        with open (file, "r") as stream:
            network = ''.join(stream.readlines())

        # load the input dictionary
        if input is not None:
            self.output_dict = input[1].copy()

        # define the patterns here
        re_input = re.compile("\[INPUT(:([\S]+))?\]")
        re_prefix = re.compile('((bottom|top|name):[ ]*)"(.*?)"')
        re_out = re.compile("\[(.+?)\][\n ]*\Z")
        re_out_layer = re.compile('layer[ ]*{[\w\s\n\W]*?top: "(.+?)"')
        re_var = re.compile("\[(\S*?)(:(\S+))?\]")
        re_eval = re.compile("\[\[(.*?)\]\]")
        re_out_dict = re.compile("\[(\S*?):(\S*)\]")

        # create empty current output
        output = []

        #-----------------------------------------------------------------------
        # define some helper functions
        def replace_input(match):
            # check type of input matching
            if len(match.groups()) > 1:
                # check if data is int or string
                if match.group(2).isdigit():
                    if int(match.group(2)) >= len(input[0]):
                        raise ValueError("Could not apply input, as item is out of range ({}) in block ({})".format(int(match.group(2)), self.name))
                    return input[0][int(match.group(2))]
                else:
                    if match.group(2) not in input[1]:
                        raise ValueError("Could not find input with referenced name ({}) in block({})!".format(match.group(2), self.name))
                    return input[1][match.group(2)]
            else:
                if len(input[0]) == 0:
                    raise ValueError("No Input Found!")
                return input[0][0]

        def add_prefix(match):
            if re.compile(re_input).search(match.group(0)) is not None:
                return match.group(0)
            else:
                return "{}\"{}/{}\"".format(match.group(1), prefix, match.group(3))

        def replace_var(match):
            # retrieve the data
            var_name = match.group(1)
            value = match.group(3) if len(match.groups()) > 2 else None

            # match to parameter
            if var_name in self.params:
                value = str(self.params[var_name])
            elif value is None:
                raise ValueError("Could not find parameter ({}) and it has no default value in block ({})!".format(var_name, self.name))
            return value

        def replace_eval(match):
            exp = match.group(1)
            exp = self._replace_string_all(re_var, exp, replace_var)
            return eval(exp)

        #-----------------------------------------------------------------------
        # generate the input

        # ADD PREFIXES (avoid inputs)
        if len(prefix) > 0:
            network = self._replace_string_all(re_prefix, network, add_prefix, True)

        # SELECT OUTPUTS
        match = re_out.search(network)
        if match is not None:
            output = match.group(1).replace(' ', '').split(',')
            network = network[:match.start()]

            # BONUS: PARSE DICT NAMES
            for i in range(len(output)):
                # match each input element against the dict
                match = re_out_dict.match(output[i])
                if match is not None:
                    # do not update if it is input directly
                    if match.group(1) == 'INPUT':
                        continue
                    # update list output
                    output[i] = match.group(2)
                    # check if right-side is input
                    match_right = re_input.match(match.group(2))
                    if match_right is None:
                        self.output_dict[match.group(1)] = "{}/{}".format(prefix, match.group(2)) if len(prefix) > 0 else match.group(2)
                    else:
                        self.output_dict[match.group(1)] = replace_input(match_right)

            # update with prefix
            if len(prefix) > 0:
                # add prefixes only for non-inputs
                output = ["{}/{}".format(prefix, out) if re_input.search(out) is None else out for out in output]
        else:
            # search the last element
            for match in re_out_layer.finditer(network):
                output = [match.group(1)]

        # BONUS: REPLACE INPUTS IN OUTPUT LIST
        for i in range(len(output)):
            output[i] = self._replace_string_all(re_input, output[i], replace_input)

        # REPLACE INPUTS
        network = self._replace_string_all(re_input, network, replace_input)

        # EVAL VARIABLES
        network = self._replace_string_all(re_eval, network, replace_eval)
        network = self._replace_string_all(re_var, network, replace_var)

        # apply the data to network
        self.network = network
        self.output = output
        return self.output, self.output_dict.copy()

    def generate(self):
        '''Returns the network string that should be added.'''
        return self.network


class Model(object):
    '''Defines the file that holds the model and all relevant information.'''

    def __init__(self, debug=False, params={}):
        '''Init the default values for the block.'''
        self.name = "DefaultNet"
        self.block = None
        self.params = params
        self._debug = debug

    def load(self, file):
        '''Loads the model based on a yaml file.'''
        # load the block
        self.block = YamlBlock(self.params, debug=self._debug)

        self.block.load(file)

        # check for the name of the network
        if self.block.name is not None:
            self.name = self.block.name

    def generate(self):
        '''Returns the generated string of the model.'''
        # write parameters (self.block.params)
        params = "# PARAMS:"
        for key in self.block.params:
            params += '\n# {:10}: {}'.format(key, str(self.block.params[key]))
        params += '\n\n# GENERATED : {:%Y-%m-%d %H-%M}'.format(datetime.datetime.now())
        return 'name: "{}"\n{}\n\n{}'.format(self.name, params, self.block.generate())

    def store(self, output):
        '''Stores the model at the given location.'''
        with open(output, 'w') as file:
            file.write(self.generate())

def generate(model, output, debug=False, **params):
    '''Generates a new caffe model from the given model.

    model (str) :
        Path to the yaml file to generate the model from
    output (str) :
        Path to the folder/file where the generated model should be stored
    debug (bool):
        Defines if debug statements should be shown
    params (args) :
        (Optional) List of parameters to apply to yaml global params (format: --NAME value)
    '''
    # load the model
    gen_model = Model(debug, params)
    gen_model.load(model)

    # safty: check the output file
    output_dir = os.path.split(output)[0]
    output_file = os.path.split(output)[1]
    if len(output_file) == 0:
        output_file = gen_model.name
    if os.path.splitext(output_file)[1] != '.prototxt':
        output_file += ".prototxt"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # generate the model
    #print(gen_model.generate())
    gen_model.store(os.path.join(output_dir, output_file))

    print("Model generated!")

def main():
    fire.Fire(generate)


if __name__ == '__main__':
    main()
