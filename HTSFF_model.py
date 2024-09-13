# -*- coding: utf-8 -*-
# @Time : 2022/7/3  16:00
# @Author : Phoenix
# @File : model_area_11_20.py
'''
describe:
'''


import torch.nn as nn
import torch
import torch.nn.functional as F

C_INPUT_DIM = 3
C_OUTPUT_DIM = 1

L_INPUT_DIM = 3
OUTPUT_DIM = 1

HID_DIM = 512
N_LAYERS = 2
ENC_DROPOUT = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Seq2Seq(nn.Module):
    def __init__(self,
                 c_input_size=C_INPUT_DIM,
                 c_output_size=C_OUTPUT_DIM,
                 l_input_size=L_INPUT_DIM,
                 output_size=OUTPUT_DIM,

                 hidden_size=HID_DIM,
                 n_layers=N_LAYERS,
                 dropout=ENC_DROPOUT):
        super(Seq2Seq, self).__init__()
        self.c_input_size = c_input_size
        self.c_output_size = c_output_size
        self.l_input_size = l_input_size
        self.output_size = output_size

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.conv1d = nn.Sequential(
            nn.Conv1d(self.c_input_size, 16, 3, stride=1),
            nn.Conv1d(16, 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 16, 3, stride=1),
            nn.Conv1d(16, self.c_output_size, 3, stride=1),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(self.l_input_size, self.hidden_size, batch_first=True,
                            dropout=self.dropout, num_layers=self.n_layers, bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(in_features=self.hidden_size*2, out_features=512),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=self.output_size)
        )

    def attention_net(self, lstm_output, final_state):
        # lstm_output : [batch_size, n_step, n_hidden * num_directions(=2)], F matrix
        # final_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        batch_size = len(lstm_output)
        # hidden = final_state.view(batch_size,-1,1)
        hidden = torch.cat((final_state[0], final_state[1]), dim=1).unsqueeze(2)
        # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)
        # attn_weights : [batch_size,n_step]
        soft_attn_weights = F.softmax(attn_weights,1)

        # context: [batch_size, n_hidden * num_directions(=2)]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)

        return context, soft_attn_weights

    def forward(self, input_x):
        conv1_step0_output = self.conv1d(input_x[:, :, 0, :]).unsqueeze(2)
        conv1_step1_output = self.conv1d(input_x[:, :, 1, :]).unsqueeze(2)
        conv1_step2_output = self.conv1d(input_x[:, :, 2, :]).unsqueeze(2)
        conv1_step3_output = self.conv1d(input_x[:, :, 3, :]).unsqueeze(2)
        conv1_step4_output = self.conv1d(input_x[:, :, 4, :]).unsqueeze(2)
        conv1_step5_output = self.conv1d(input_x[:, :, 5, :]).unsqueeze(2)
        conv1_step6_output = self.conv1d(input_x[:, :, 6, :]).unsqueeze(2)
        conv1_step7_output = self.conv1d(input_x[:, :, 7, :]).unsqueeze(2)
        conv1_step8_output = self.conv1d(input_x[:, :, 8, :]).unsqueeze(2)
        conv1_step9_output = self.conv1d(input_x[:, :, 9, :]).unsqueeze(2)


        output = torch.cat((conv1_step0_output, conv1_step1_output, conv1_step2_output, conv1_step3_output,
                            conv1_step4_output, conv1_step5_output, conv1_step6_output, conv1_step7_output,
                            conv1_step8_output, conv1_step9_output), 2)


        output = output.squeeze(1)

        output, (final_hidden_state, final_cell_state) = self.lstm(output)

        # #注意力机制
        # attn_output, attention = self.attention_net(output, final_hidden_state)
        # out = self.linear(attn_output).squeeze()

        out = self.linear(output[:, -1:, :]).squeeze()

        return out



model = Seq2Seq()

if __name__ == "__main__":
    matrix_data = torch.randn((256, 3, 15, 11))
    out = model(matrix_data)
    print(model)
    print(out.size())