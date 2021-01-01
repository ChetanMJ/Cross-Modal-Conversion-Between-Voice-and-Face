from configuration import *

class ConvBlock(nn.Module):

    def __init__(self, input_channel_size, output_channel_size,kernel_size=1, stride=1, groups=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel_size, 
                               output_channel_size, kernel_size=kernel_size, stride=stride, 
                               padding=padding, bias=False, groups=groups)
        self.bn1 = nn.BatchNorm2d(output_channel_size)
        self.LRelu = nn.LeakyReLU(inplace=True)
        
    def forward(self, x):
        out =   self.LRelu(self.bn1(self.conv1(x)))
        return out
        

class DeConvBlock(nn.Module):

    def __init__(self, input_channel_size, output_channel_size,kernel_size=1, stride=1,padding=0):
        super(DeConvBlock, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(input_channel_size, 
                               output_channel_size, kernel_size=kernel_size, stride=stride, 
                               padding=padding)
        self.bn1 = nn.BatchNorm2d(output_channel_size)
        self.softplus = nn.Softplus()
        
    def forward(self, x):
        out =   self.softplus(self.bn1(self.deconv1(x)))
        return out
        
class FaceEncoder(nn.Module):
    ''' This the encoder part of VAE
    '''
    def __init__(self, input_dim, n_classes):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()
        
        self.n_classes = n_classes
      
        self.FaceEncoder_Layers = []
        
        self.FaceEncoder_Layers.append(nn.Conv2d(in_channels=3, 
                               out_channels=32, kernel_size=6, stride=2, 
                               padding=0, bias=False, groups=1))
        
        self.FaceEncoder_Layers.append(nn.LeakyReLU(inplace=True))
        
        self.FaceEncoder_Layers.append(ConvBlock(input_channel_size=32, output_channel_size=64,kernel_size=6, stride=2))
        self.FaceEncoder_Layers.append(ConvBlock(input_channel_size=64, output_channel_size=128,kernel_size=4, stride=2))
        self.FaceEncoder_Layers.append(ConvBlock(input_channel_size=128, output_channel_size=128,kernel_size=4, stride=2, padding=2))
        self.FaceEncoder_Layers.append(ConvBlock(input_channel_size=128, output_channel_size=256,kernel_size=2, stride=2, padding=1))
        self.FaceEncoder_Layers.append(ConvBlock(input_channel_size=256, output_channel_size=256,kernel_size=2, stride=2, padding=0))
        self.FaceEncoder_Layers = nn.Sequential(*self.FaceEncoder_Layers)
        
        
        self.FaceEncoder_Layers2 = []
        self.FaceEncoder_Layers2.append(nn.Linear(256, 256))
        self.FaceEncoder_Layers2.append(nn.LeakyReLU(inplace=True))
        self.FaceEncoder_Layers2.append(nn.Linear(256, 16))
        self.FaceEncoder_Layers2.append(nn.LeakyReLU(inplace=True))
        self.FaceEncoder_Layers2 = nn.Sequential(*self.FaceEncoder_Layers2)
        
       
        
        self.mu_layer = nn.Conv2d(in_channels=16, 
                     out_channels=8, kernel_size=1, stride=1, 
                     padding=0, bias=False, groups=1)
        self.var_layer = nn.Conv2d(in_channels=16, 
                      out_channels=8, kernel_size=1, stride=1, 
                      padding=0, bias=False, groups=1)
               
    def forward(self, x):
        
        out = self.FaceEncoder_Layers(x)
        out = out.transpose(1,3)       
        out = self.FaceEncoder_Layers2(out)
        out = out.transpose(1,3)
        mean = self.mu_layer(out)
        log_var = self.var_layer(out)

        return mean, log_var

class FaceDecoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self, latent_dim, n_classes):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.batch_size = batch_size
        
        self.lin = nn.Linear(1, 128)
        
        self.FaceDecoder_Layers = []
        self.FaceDecoder_Layers.append(nn.Linear(self.latent_dim*2, 128))
        self.FaceDecoder_Layers.append(nn.Softplus())
        self.FaceDecoder_Layers.append(nn.Linear(128, 2048))
        self.FaceDecoder_Layers.append(nn.Softplus())
        self.FaceDecoder_Layers = nn.Sequential(*self.FaceDecoder_Layers)
        
        self.FaceDecoder_Layers1 = []
        self.FaceDecoder_Layers1.append(nn.Linear(136, 1))
        self.FaceDecoder_Layers1.append(nn.Softplus())
        self.FaceDecoder_Layers1 = nn.Sequential(*self.FaceDecoder_Layers1)

        
        self.FaceDecoder_Layers21 = DeConvBlock(input_channel_size=192, output_channel_size=128, kernel_size=3, stride=2,padding=0)
        self.FaceDecoder_Layers22 = DeConvBlock(input_channel_size=136, output_channel_size=128, kernel_size=6, stride=2)
        self.FaceDecoder_Layers23 = DeConvBlock(input_channel_size=128, output_channel_size=64, kernel_size=6, stride=2)
        self.FaceDecoder_Layers24 = DeConvBlock(input_channel_size=72, output_channel_size=32, kernel_size=6, stride=2)
        self.FaceDecoder_Layers25 = DeConvBlock(input_channel_size=32, output_channel_size=16, kernel_size=6, stride=2)
        self.FaceDecoder_Layers26 = nn.Conv2d(in_channels=24,out_channels=6, kernel_size=6, stride=1)

        
        
        self.mu_conv = nn.Conv2d(in_channels=6,out_channels=3, kernel_size=8, stride=3)
        self.var_conv = nn.Conv2d(in_channels=6,out_channels=3, kernel_size=8, stride=3)
        
        ##bcast linears
        self.lin1 = nn.Linear(128,2048)
        self.lin2 = nn.Linear(128,81)
        self.lin3 = nn.Linear(128,2304)
        self.lin4 = nn.Linear(48,204)
        self.lin5 = nn.Linear(48,204)

    def forward(self, x, y):
        
        batch_size = x.size()[0]
        
        y1 = self.lin(y)
        x1 = torch.cat((x,y1), dim=1)    
        
        z1 = x1.transpose(1,3)      
        
        out = self.FaceDecoder_Layers(z1)
        out = torch.cat((out,self.lin1(x)), dim=1)
        out = out.transpose(1,3)
        out = self.FaceDecoder_Layers1(out)
        out = out.view(batch_size,128,4,4)
        x1 = x.view(batch_size,64,4,4)
        out = torch.cat((out,x1), dim=1)
        
        ##gen_img = self.FaceDecoder_Layers2(out)
        
        out = self.FaceDecoder_Layers21(out)       
        x2 = self.lin2(x)
        x2 = x2.view(batch_size,8,9,9)    
        out = torch.cat((out,x2), dim=1)
        
        out = self.FaceDecoder_Layers22(out)        
        out = self.FaceDecoder_Layers23(out)
        x3 = self.lin3(x)
        x3 = x3.view(batch_size,8,48,48)    
        out = torch.cat((out,x3), dim=1)
        
        out = self.FaceDecoder_Layers24(out)
        out = self.FaceDecoder_Layers25(out)

        x4 = self.lin4(x3)
        x5 = self.lin5(x4.transpose(2,3))
        out = torch.cat((out,x5), dim=1)

        gen_img = self.FaceDecoder_Layers26(out)
        
        
        img_mu = self.mu_conv(gen_img)
        img_var = self.var_conv(gen_img)
        
        return img_mu, img_var
 
class UtteranceEncoder(nn.Module):
     ''' This the decoder part of VAE
     '''
     def __init__(self, input_dim):
         '''
         Args:
             latent_dim: A integer indicating the latent size.
             hidden_dim: A integer indicating the size of hidden dimension.
             output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
             n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
         '''
         super().__init__()
         self.input_dim = input_dim
         
         self.conv1 = nn.Conv2d(self.input_dim, 8, (3,9), (1,1), padding=(1, 4))
         self.conv1_bn = nn.BatchNorm2d(8)
         self.conv1_gated = nn.Conv2d(self.input_dim, 8, (3,9), (1,1), padding=(1, 4))
         self.conv1_gated_bn = nn.BatchNorm2d(8)
         self.conv1_sigmoid = nn.Sigmoid()
 
         self.conv2 = nn.Conv2d(8, 16, (4,8), (2,2), padding=(1, 3))
         self.conv2_bn = nn.BatchNorm2d(16)
         self.conv2_gated = nn.Conv2d(8, 16, (4,8), (2,2), padding=(1, 3))
         self.conv2_gated_bn = nn.BatchNorm2d(16)
         self.conv2_sigmoid = nn.Sigmoid()
 
         self.conv3 = nn.Conv2d(16, 16, (4,8), (2,2), padding=(1, 3))
         self.conv3_bn = nn.BatchNorm2d(16)
         self.conv3_gated = nn.Conv2d(16, 16, (4,8), (2,2), padding=(1, 3))
         self.conv3_gated_bn = nn.BatchNorm2d(16)
         self.conv3_sigmoid = nn.Sigmoid() 
 
         self.conv4_mu = nn.Conv2d(16, 16//2, (9,5), (9,1), padding=(1, 2))
         self.conv4_logvar = nn.Conv2d(16, 16//2, (9,5), (9,1), padding=(1, 2))
         
     def forward(self, x):
  
         h1_ = self.conv1_bn(self.conv1(x))
         h1_gated = self.conv1_gated_bn(self.conv1_gated(x))
         h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 
        
         h2_ = self.conv2_bn(self.conv2(h1))
         h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
         h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated))
         
         h3_ = self.conv3_bn(self.conv3(h2))
         h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
         h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated)) 
         
         h4_mu = self.conv4_mu(h3)
         h4_logvar = self.conv4_logvar(h3)
        
         return h4_mu, h4_logvar

class UtteranceDecoder(nn.Module):
    ''' This the decoder part of VAE
    '''
    def __init__(self, input_dim, face_latent_dim):
        '''
        Args:
            latent_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden dimension.
            output_dim: A integer indicating the size of output (in case of MNIST 28 * 28).
            n_classes: A integer indicating the number of classes. (dimension of one-hot representation of labels)
        '''
        super().__init__()
        self.input_dim = input_dim
        self.face_latent_dim = face_latent_dim
        
        # Decoder

        self.upconv1 = nn.ConvTranspose2d(self.input_dim + self.face_latent_dim, 16, (9,5), (9,1), padding=(0, 2))
        self.upconv1_bn = nn.BatchNorm2d(16)
        self.upconv1_gated = nn.ConvTranspose2d(self.input_dim + self.face_latent_dim, 16, (9,5), (9,1), padding=(0, 2))
        self.upconv1_gated_bn = nn.BatchNorm2d(16)
        self.upconv1_sigmoid = nn.Sigmoid()
        
        self.upconv2 = nn.ConvTranspose2d(16+self.face_latent_dim, 16, (4,8), (2,2), padding=(1, 3))
        self.upconv2_bn = nn.BatchNorm2d(16)
        self.upconv2_gated = nn.ConvTranspose2d(16+self.face_latent_dim, 16, (4,8), (2,2), padding=(1, 3))
        self.upconv2_gated_bn = nn.BatchNorm2d(16)
        self.upconv2_sigmoid = nn.Sigmoid()
        
        self.upconv3 = nn.ConvTranspose2d(16+self.face_latent_dim, 8, (4,8), (2,2), padding=(1, 3))
        self.upconv3_bn = nn.BatchNorm2d(8)
        self.upconv3_gated = nn.ConvTranspose2d(16+self.face_latent_dim, 8, (4,8), (2,2), padding=(1, 3))
        self.upconv3_gated_bn = nn.BatchNorm2d(8)
        self.upconv3_sigmoid = nn.Sigmoid()
        
        self.upconv4_mu = nn.ConvTranspose2d(8+self.face_latent_dim, 2//2, (3,9), (1,1), padding=(1, 4))
        self.upconv4_logvar = nn.ConvTranspose2d(8+self.face_latent_dim, 2//2, (3,9), (1,1), padding=(1, 4))
        
        ##linear layers for face latent broadcasting
        self.N = 1024
        self.bcast_linear1 = nn.Linear(1,int(self.N/4))  ##n/4 = 256 where n = 1024 frames
        self.bcast_linear2 = nn.Linear(1,9)
        self.bcast_linear3 = nn.Linear(9,18)
        self.bcast_linear4 = nn.Linear(int(self.N/4),int(self.N/2))
        self.bcast_linear5 = nn.Linear(18,36)
        self.bcast_linear6 = nn.Linear(int(self.N/2),self.N)
        
        
    def forward(self, z, face_latent):
        
        
        face_latent_b1 = self.bcast_linear1(face_latent)
        
        h5_ = self.upconv1_bn(self.upconv1(torch.cat((z, face_latent_b1), dim=1)))
        h5_gated = self.upconv1_gated_bn(self.upconv1(torch.cat((z, face_latent_b1), dim=1)))
        h5 = torch.mul(h5_, self.upconv1_sigmoid(h5_gated)) 
         
        face_latent_b1 = face_latent_b1.transpose(2,3)
        face_latent_b2 = self.bcast_linear2(face_latent_b1)
        face_latent_b2 = face_latent_b2.transpose(2,3)
        
        h6_ = self.upconv2_bn(self.upconv2(torch.cat((h5, face_latent_b2), dim=1)))
        h6_gated = self.upconv2_gated_bn(self.upconv2(torch.cat((h5, face_latent_b2), dim=1)))
        h6 = torch.mul(h6_, self.upconv2_sigmoid(h6_gated)) 
        
        face_latent_b2 = face_latent_b2.transpose(2,3)
        face_latent_b3 = self.bcast_linear3(face_latent_b2)
        face_latent_b3 = face_latent_b3.transpose(2,3)
        face_latent_b3 = self.bcast_linear4(face_latent_b3)                  
                              
        h7_ = self.upconv3_bn(self.upconv3(torch.cat((h6, face_latent_b3), dim=1)))
        h7_gated = self.upconv3_gated_bn(self.upconv3(torch.cat((h6, face_latent_b3), dim=1)))
        h7 = torch.mul(h7_, self.upconv3_sigmoid(h7_gated)) 

        
        face_latent_b3 = face_latent_b3.transpose(2,3)
        face_latent_b3 = self.bcast_linear5(face_latent_b3)
        face_latent_b3 = face_latent_b3.transpose(2,3)
        face_latent_b4 = self.bcast_linear6(face_latent_b3)
                              
        
        h8_mu = self.upconv4_mu(torch.cat((h7, face_latent_b4), dim=1))
        h8_logvar = self.upconv4_logvar(torch.cat((h7, face_latent_b4), dim=1))
        
        return h8_mu, h8_logvar

class VoiceEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        self.conv1 = nn.Conv2d(1, 32, (3,9), (1,1), padding=(1, 4))
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_gated = nn.Conv2d(1, 32, (3,9), (1,1), padding=(1, 4))
        self.conv1_gated_bn = nn.BatchNorm2d(32)
        self.conv1_sigmoid = nn.Sigmoid()
        
        self.conv2 = nn.Conv2d(32, 64, (4,8), (2,2), padding=(1, 3))
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_gated = nn.Conv2d(32, 64, (4,8), (2,2), padding=(1, 3))
        self.conv2_gated_bn = nn.BatchNorm2d(64)
        self.conv2_sigmoid = nn.Sigmoid()
        
        self.conv3 = nn.Conv2d(64, 128, (4,8), (2,2), padding=(1, 3))
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_gated = nn.Conv2d(64, 128, (4,8), (2,2), padding=(1, 3))
        self.conv3_gated_bn = nn.BatchNorm2d(128)
        self.conv3_sigmoid = nn.Sigmoid()
        
        self.conv4 = nn.Conv2d(128, 128, (4,8), (2,2), padding=(1, 3))
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv4_gated = nn.Conv2d(128, 128, (4,8), (2,2), padding=(1, 3))
        self.conv4_gated_bn = nn.BatchNorm2d(128)
        self.conv4_sigmoid = nn.Sigmoid()
        
        self.conv5 = nn.Conv2d(128, 128, (4,5), (4,1), padding=(1, 5))
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv5_gated = nn.Conv2d(128, 128, (4,5), (4,1), padding=(1, 5))
        self.conv5_gated_bn = nn.BatchNorm2d(128)
        self.conv5_sigmoid = nn.Sigmoid()
        
        self.conv6 = nn.Conv2d(128, 64, (1,5), (1,1), padding=(1, 5))
        self.conv6_bn = nn.BatchNorm2d(64)
        self.conv6_gated = nn.Conv2d(128, 64, (1,5), (1,1), padding=(1, 5))
        self.conv6_gated_bn = nn.BatchNorm2d(64)
        self.conv6_sigmoid = nn.Sigmoid()
        
        self.conv7 = nn.Conv2d(64, 16, (1,5), (1,1), padding=(1, 5))
        self.conv7_bn = nn.BatchNorm2d(16)
        self.conv7_gated = nn.Conv2d(64, 16, (1,5), (1,1), padding=(1, 5))
        self.conv7_gated_bn = nn.BatchNorm2d(16)
        self.conv7_sigmoid = nn.Sigmoid()

        self.conv8 = nn.Conv2d(16, 8, (1,5), (1,1), padding=(1, 5))

        
        self.lin1_mu = nn.Linear(7,1)
        self.lin2_mu = nn.Linear(152,128)
        
        self.lin1_var = nn.Linear(7,1)
        self.lin2_var = nn.Linear(152,128)
        
        
    def forward(self, x):
        
        h1_ = self.conv1_bn(self.conv1(x))
        h1_gated = self.conv1_gated_bn(self.conv1_gated(x))
        h1 = torch.mul(h1_, self.conv1_sigmoid(h1_gated)) 
        
        h2_ = self.conv2_bn(self.conv2(h1))
        h2_gated = self.conv2_gated_bn(self.conv2_gated(h1))
        h2 = torch.mul(h2_, self.conv2_sigmoid(h2_gated))
        
        h3_ = self.conv3_bn(self.conv3(h2))
        h3_gated = self.conv3_gated_bn(self.conv3_gated(h2))
        h3 = torch.mul(h3_, self.conv3_sigmoid(h3_gated))

        h4_ = self.conv4_bn(self.conv4(h3))
        h4_gated = self.conv4_gated_bn(self.conv4_gated(h3))
        h4 = torch.mul(h4_, self.conv4_sigmoid(h4_gated))
        
        h5_ = self.conv5_bn(self.conv5(h4))
        h5_gated = self.conv5_gated_bn(self.conv5_gated(h4))
        h5 = torch.mul(h5_, self.conv5_sigmoid(h5_gated))
        
        h6_ = self.conv6_bn(self.conv6(h5))
        h6_gated = self.conv6_gated_bn(self.conv6_gated(h5))
        h6 = torch.mul(h6_, self.conv6_sigmoid(h6_gated))
        
        h7_ = self.conv7_bn(self.conv7(h6))
        h7_gated = self.conv7_gated_bn(self.conv7_gated(h6))
        h7 = torch.mul(h7_, self.conv7_sigmoid(h7_gated))
        
        h8 = self.conv8(h7) 
        
        h8 = h8.transpose(2,3)
        
        
        h9_mu = self.lin1_mu(h8)
        h9_mu = h9_mu.transpose(2,3)
        h10_mu = self.lin2_mu(h9_mu)
        
        
        h9_var = self.lin1_var(h8)
        h9_var = h9_var.transpose(2,3)
        h10_var = self.lin2_var(h9_var)        
        
        
        return h10_mu, h10_var
 
class Face_Discriminator(nn.Module):
     """Discriminator network with PatchGAN."""
     def __init__(self, image_size=64, conv_dim=64, c_dim=4, repeat_num=6):
         super(Face_Discriminator, self).__init__()
         layers = []
         layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
         layers.append(nn.LeakyReLU(0.01))
 
         curr_dim = conv_dim
         for i in range(1, repeat_num):
             layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
             layers.append(nn.LeakyReLU(0.01))
             curr_dim = curr_dim * 2
 
         kernel_size = int(image_size / np.power(2, repeat_num))
         self.main = nn.Sequential(*layers)
         self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
         self.conv2 = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
         #self.sig = nn.Sigmoid()
         #self.sig_2 = nn.Sigmoid()
         
     def forward(self, x):
         h = self.main(x)
         out_src = self.conv1(h)
         out_cls = self.conv2(h)
         out_cls = out_cls.view(out_cls.size(0), out_cls.size(1))
         return F.softmax(out_cls, dim=1)

class Voice_Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, label_num):
        super(Voice_Discriminator, self).__init__()
        
        self.label_num = label_num
        
        self.ac_conv1 = nn.Conv2d(1, 8, (4,4), (2,2), padding=(1, 1))
        self.ac_conv1_bn = nn.BatchNorm2d(8)
        self.ac_conv1_gated = nn.Conv2d(1, 8, (4,4), (2,2), padding=(1, 1))
        self.ac_conv1_gated_bn = nn.BatchNorm2d(8)
        self.ac_conv1_sigmoid = nn.Sigmoid()
        
        self.ac_conv2 = nn.Conv2d(8, 16, (4,4), (2,2), padding=(1, 1))
        self.ac_conv2_bn = nn.BatchNorm2d(16)
        self.ac_conv2_gated = nn.Conv2d(8, 16, (4,4), (2,2), padding=(1, 1))
        self.ac_conv2_gated_bn = nn.BatchNorm2d(16)
        self.ac_conv2_sigmoid = nn.Sigmoid()
        
        self.ac_conv3 = nn.Conv2d(16, 32, (4,4), (2,2), padding=(1, 1))
        self.ac_conv3_bn = nn.BatchNorm2d(32)
        self.ac_conv3_gated = nn.Conv2d(16, 32, (4,4), (2,2), padding=(1, 1))
        self.ac_conv3_gated_bn = nn.BatchNorm2d(32)
        self.ac_conv3_sigmoid = nn.Sigmoid()
        
        self.ac_conv4 = nn.Conv2d(32, 16, (4,4), (2,2), padding=(1, 1))
        self.ac_conv4_bn = nn.BatchNorm2d(16)
        self.ac_conv4_gated = nn.Conv2d(32, 16, (4,4), (2,2), padding=(1, 1))
        self.ac_conv4_gated_bn = nn.BatchNorm2d(16)
        self.ac_conv4_sigmoid = nn.Sigmoid()
        
        self.ac_conv5 = nn.Conv2d(16, self.label_num, (1,4), (1,2), padding=(0, 1))
        self.ac_lin1 = nn.Linear(32,1)
        self.ac_conv6 = nn.Conv2d(16, 1, (1,4), (1,2), padding=(0, 1))
        self.ac_lin2 = nn.Linear(32,1)
        #self.sig1 = nn.Sigmoid()
        #self.sig2 = nn.Sigmoid()
        
    def forward(self, x):
        x = x[:,:,:16]
   
        h9_ = self.ac_conv1_bn(self.ac_conv1(x))
        h9_gated = self.ac_conv1_gated_bn(self.ac_conv1_gated(x))
        h9 = torch.mul(h9_, self.ac_conv1_sigmoid(h9_gated))
        
        h10_ = self.ac_conv2_bn(self.ac_conv2(h9))
        h10_gated = self.ac_conv2_gated_bn(self.ac_conv2_gated(h9))
        h10 = torch.mul(h10_, self.ac_conv2_sigmoid(h10_gated))
        
        h11_ = self.ac_conv3_bn(self.ac_conv3(h10))
        h11_gated = self.ac_conv3_gated_bn(self.ac_conv3_gated(h10))
        h11 = torch.mul(h11_, self.ac_conv3_sigmoid(h11_gated))
        
        h12_ = self.ac_conv4_bn(self.ac_conv4(h11))
        h12_gated = self.ac_conv4_gated_bn(self.ac_conv4_gated(h11))
        h12 = torch.mul(h12_, self.ac_conv4_sigmoid(h12_gated))
        
        h13_ = F.softmax(self.ac_conv5(h12), dim=1)
        h13 = torch.prod(h13_, dim=-1, keepdim=True)
       
        return h13.view(-1, self.label_num)


class CrossModal(nn.Module):
    def __init__(self, lambda1, lambda2, xx1):
        super().__init__()
        self.xx = xx1
        self.lambda_1 = lambda1
        self.lambda_2 = lambda2
        self.Utterance_Encoder = UtteranceEncoder(1)
        self.Utterance_Decoder = UtteranceDecoder(8,8)
        self.Face_Encoder = FaceEncoder(3,2)
        self.Face_Decoder = FaceDecoder(8,2)
        self.Voice_Encoder = VoiceEncoder(1)
        self.Face_Classifier = Face_Discriminator(c_dim=4)
        self.Voice_Classifier = Voice_Discriminator(4)
        self.L2_loss1 = nn.MSELoss()
        self.L2_loss2 = nn.MSELoss()
        
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
    
    
    def forward(self, utterance, face ):
        batch_size = utterance.size()[0]
        UE_mu, UE_var = self.Utterance_Encoder(utterance)
        FE_mu, FE_var = self.Face_Encoder(face)
        reparam_UE = self.reparameterize(UE_mu, UE_var)
        reparam_FE = self.reparameterize(FE_mu, FE_var)
        UD_mu, UD_var = self.Utterance_Decoder(reparam_UE,reparam_FE)
        recon_voice = self.reparameterize(UD_mu, UD_var)
        voice_label_p = self.Voice_Classifier(recon_voice)
        
        VE_mu, VE_var = self.Voice_Encoder(recon_voice)
        reparam_VE = self.reparameterize(VE_mu, VE_var)
        recon_image_mu, recon_image_var = self.Face_Decoder(reparam_VE,self.xx[:batch_size, :,:,:])       
        recon_image = self.reparameterize(recon_image_mu, recon_image_var)
        image_label_p = self.Face_Classifier(recon_image)
        
        return recon_voice, recon_image, UE_mu, UE_var, FE_mu, FE_var, voice_label_p, image_label_p
    
    
    def calculate_loss(self, utterance, face, label):
        
        self.real_label = label
        
        re_voice, re_face, u_mu, u_var, f_mu, f_var, re_voice_label_p, re_face_label_p = self.forward(utterance, face)
        
        t_label_voice = self.Voice_Classifier(utterance)
        t_label_face = self.Face_Classifier(face)
        
        L1_voice = torch.sum(torch.abs(re_voice - utterance))
        L1_face = torch.sum(torch.abs(re_face - face))
        
        #L2_voice = self.L2_loss1(re_voice, utterance)
        #L2_face = self.L2_loss2(re_face, face)
        
        #rec_loss_voice = F.binary_cross_entropy_with_logits(re_voice, utterance)
        #rec_loss_face = F.binary_cross_entropy_with_logits(re_face, face)
        
        KLD_voice = -0.5 * torch.sum(1 + u_var - u_mu.pow(2) - u_var.exp())
        KLD_face = -0.5 * torch.sum(1 + f_var - f_mu.pow(2) - f_var.exp())
        
        AC_1_voice = self.lambda_1 * F.binary_cross_entropy(re_voice_label_p, self.real_label) 
        AC_2_voice = self.lambda_2 * F.binary_cross_entropy(t_label_voice, self.real_label)

        #print(re_face_label_p)
        AC_1_face = self.lambda_1 * F.binary_cross_entropy(re_face_label_p, self.real_label) 
        AC_2_face = self.lambda_2 * F.binary_cross_entropy(t_label_face, self.real_label)
        
        #+ AC_1_voice + AC_2_voice
        
        return L1_voice + L1_face + KLD_voice + KLD_face + AC_1_face + AC_2_face 
 
 