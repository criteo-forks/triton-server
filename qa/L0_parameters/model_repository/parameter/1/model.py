# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import triton_python_backend_utils as pb_utils
import numpy as np
import json


class TritonPythonModel:

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [{'name': 'INPUT0', 'data_type': 'TYPE_FP32', 'dims': [1]}]
        outputs = [{
            'name': 'key',
            'data_type': 'TYPE_STRING',
            'dims': [-1]
        }, {
            'name': 'value',
            'data_type': 'TYPE_STRING',
            'dims': [-1]
        }]

        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        auto_complete_model_config.set_max_batch_size(0)
        return auto_complete_model_config

    def execute(self, requests):
        # A simple model that puts the parameters in the in the request in the
        # output.
        responses = []
        for request in requests:
            parameters = json.loads(request.parameters())
            keys = []
            values = []
            for key, value in parameters.items():
                keys.append(key)
                values.append(value)
            key_output = pb_utils.Tensor("key", np.asarray(keys, dtype=object))
            value_output = pb_utils.Tensor("value",
                                           np.asarray(values, dtype=object))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[key_output, value_output])
            responses.append(inference_response)

        return responses
