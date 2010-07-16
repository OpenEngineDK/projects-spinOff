// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------

#include "MRIModule.h"
#include <Meta/CUDA.h>

#include <Resources/Texture2D.h>

#include <Logging/Logger.h>
#include "../FFT_wrap/FFT.hcu"
#include "MRI.hcu"

namespace OpenEngine {
namespace Science {

using namespace Devices;

MRIModule::MRIModule(ITextureResourcePtr img) 
    : img(img)
    , outputTexture(EmptyTextureResource::Create(img->GetWidth(), 
                                                 img->GetHeight(), 
                                                 24))
    , inverseTexture(EmptyTextureResource::Create(img->GetWidth(), 
                                                  img->GetHeight(), 
                                                  8))
    , running(false)
    , b0(1.0)
    , spinPackets(NULL)
    , eq(NULL)
    , idx(100)
 {
 }

void MRIModule::Handle(ProcessEventArg arg) {
    if (running) {
        unsigned int w = img->GetWidth();
        unsigned int h = img->GetHeight();
        float dt = arg.approx * 0.000001;

        logger.info << "running kernel (dt: " << dt << "sec)" << logger.end;
        MRI_step(dt, spinPackets, eq, img->GetWidth(), img->GetHeight(), b0);

        float* data = (float*)malloc(sizeof(float3) * w * h);
        cudaMemcpy(data, spinPackets, w * h * sizeof(float3), cudaMemcpyDeviceToHost);
        unsigned int index = idx;
        Vector<3,float> magnet(data[index*3], data[index*3+1], data[index*3+2]);
        logger.info << "reading index: " << index << " with value: " << magnet << logger.end;
        free(data);
    }
}

void MRIModule::Handle(InitializeEventArg arg) {
    INITIALIZE_CUDA();
    //logger.info << PRINT_CUDA_DEVICE_INFO() << logger.end;

    unsigned int w = img->GetWidth();
    unsigned int h = img->GetHeight();

    float* data = (float*)malloc(sizeof(float3) * w * h);
    float* meq  = (float*)malloc(sizeof(float) * w * h);
    float scale = 1.0;

    UCharTexture2D* input = dynamic_cast<UCharTexture2D*>(img.get()); 

    for (unsigned int i=0; i < w;i++) {
        for (unsigned int j=0; j < h;j++) {
             Vector<4,unsigned char> pixel = input->GetPixelValues(i,j);
             float pix = (0.3*pixel[0] + 0.59*pixel[1] + 0.11*pixel[2]);
             pix /= 255;
        
             data[(i*h+j)*3]   = 0.0;
             data[(i*h+j)*3+1] = 0.0;
             data[(i*h+j)*3+2] = scale*pix;

             meq[i*h+j] = scale*pix;
        }
    }

    cudaMalloc((void**)&spinPackets, sizeof(float3) * w * h);        
    cudaMalloc((void**)&eq, sizeof(float) * w * h);        
    cudaMemcpy(spinPackets, data, w * h * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(eq, meq, w * h * sizeof(float), cudaMemcpyHostToDevice);
    free(data);    
    free(meq);    
}

void MRIModule::Handle(DeinitializeEventArg arg) {
    
}

void MRIModule::Handle(KeyboardEventArg arg) {
    if (arg.sym == KEY_k && arg.type == EVENT_RELEASE) {
        
        unsigned int w = img->GetWidth();
        unsigned int h = img->GetHeight();

        uint2 dims = make_uint2(w,h);

        cuFloatComplex* data = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * w * h);

        UCharTexture2D* input = dynamic_cast<UCharTexture2D*>(img.get()); 

        //logger.info << "input = " << input << logger.end;

        for (unsigned int i=0;i<w;i++) {
            for (unsigned int j=0;j<h;j++) {
                
                Vector<4,unsigned char> pixel = input->GetPixelValues(i,j);

                float pix = (0.3*pixel[0] + 0.59*pixel[1] + 0.11*pixel[2]);
                pix /= 255;

                data[i*h+j] = make_cuFloatComplex(pix,0);
            }
        }

        MRI_test(data);

        cuFloatComplex* devData;
        cudaMalloc((void**)&devData, sizeof(cuFloatComplex) * w * h);
        
        cudaMemcpy(devData, data, w * h * sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

        bool b = I2K_ALL(devData, dims);
        logger.info << "I2K_ALL = " << b << logger.end;

        cudaMemcpy(data, devData, w * h * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

        for (unsigned int i=0;i<w;i++) {
            for (unsigned int j=0;j<h;j++) {
                cuFloatComplex c = data[i*h+j];
                char intens = 255*sqrt(c.x*c.x + c.y*c.y)/3;
                (*outputTexture)(i,j,0) = intens;
                (*outputTexture)(i,j,1) = intens;
                (*outputTexture)(i,j,2) = intens;


                // (*outputTexture)(i,j,0) = 0;//255*sqrt(c.x*c.x + c.y*c.y);
                // (*outputTexture)(i,j,1) = 255*c.x;
                // (*outputTexture)(i,j,2) = 255*c.y;
            }
        }
        outputTexture->RebindTexture();

        // k 2 i

        K2I_ALL(devData, dims);

        cudaMemcpy(data, devData, w * h * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

        for (unsigned int i=0;i<w;i++) {
            for (unsigned int j=0;j<h;j++) {
                cuFloatComplex c = data[i*h+j];                
                (*inverseTexture)(i,j) = 255*sqrt(c.x*c.x + c.y*c.y);
                //(*inverseTexture)(i,j) = 255*c.x;
            }
        }
        inverseTexture->RebindTexture();

        cudaFree(devData);
        free(data);
    } 
}

using namespace Utils::Inspection;

#define MRI_INSPECTION(type, field, _name)                              \
    do {                                                                \
    RWValueCall<MRIModule, type> *v                                     \
    = new RWValueCall<MRIModule, type>(*this,                           \
                                       &MRIModule::Get##field,          \
                                       &MRIModule::Set##field);         \
    v->name = _name;                                                    \
    values.push_back(v);                                                \
    } while (0)

ValueList MRIModule::Inspection() {
    ValueList values;

    MRI_INSPECTION(bool, Running, "running"); // simulation toggle
    MRI_INSPECTION(float, B0, "B0 (Tesla)");  // B0 field strength
    MRI_INSPECTION(unsigned int, Index, "test index");  // 

    return values;
}


} // NS Science

namespace Utils {
namespace Inspection {
    
ValueList Inspect(Science::MRIModule *mri) {
    return mri->Inspection();
}
    
}
}

} // NS OpenEngine
