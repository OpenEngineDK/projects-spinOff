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
#include <stdint.h>

#include <Geometry/Line.h>

namespace OpenEngine {
namespace Science {

using namespace Devices;
using namespace Geometry;

MRIModule::MRIModule(ITextureResourcePtr img) 
    : img(img)
    , outputTexture(EmptyTextureResource::Create(img->GetWidth(), 
                                                 img->GetHeight(), 
                                                 24))
    , inverseTexture(EmptyTextureResource::Create(img->GetWidth(), 
                                                  img->GetHeight(), 
                                                  8))
    , testOutputTexture(EmptyTextureResource::Create(img->GetWidth(), 
                                                     img->GetHeight(), 
                                                     24))
    , descaledOutputTexture(EmptyTextureResource::Create(4,4,24))
    , running(false)
    , fid(false)
    , b0(.5)
    , lab_spins(NULL)
    , ref_spins(NULL)
    , props(NULL)
    , idx(99747)
 {
 }

void MRIModule::Handle(Renderers::RenderingEventArg arg) {
    unsigned int w = 4;
    unsigned int h = 4;
    float size = 5.0;
    float space = 2.0;

    arg.renderer.ApplyViewingVolume(*arg.canvas.GetViewingVolume());

    for (unsigned int i=0;i<w;i++) {
        for (unsigned int j=0;j<h;j++) {                
            Vector<3,float> dir = descaledVectors[i][j]*size;
            //logger.info << "lvec: " << dir.GetLength() << logger.end;
            arg.renderer.DrawLine(Line(Vector<3,float>(i,j,0.0)*space, Vector<3,float>(i*space+dir[0], j*space+dir[1],dir[2])), Vector<3,float>(1.0,0.0,0.0), 2.0);
        }
    }
}

void MRIModule::Handle(ProcessEventArg arg) {
    if (running) {
        unsigned int w = img->GetWidth();
        unsigned int h = img->GetHeight();
        float timeScale = 0.000001;
        float dt = arg.approx * 0.000001 * timeScale;
        dt = 1e-7;
        //dt = arg.approx * 1e-13;
        logger.info << "running kernel (dt: " << dt << "sec)" << logger.end;

        float3 b = make_float3(0.0,0.0,b0);
        if (fid) {
            logger.info << "FID" << logger.end;
            b += make_float3(Math::PI*0.5,0.0,0.0);
            fid = false;
        }
        MRI_step(dt, (float3*)lab_spins, (float3*)ref_spins, props, img->GetWidth(), img->GetHeight(), b);

        float* data = (float*)malloc(sizeof(float3) * w * h);
        cudaMemcpy(data, lab_spins, w * h * sizeof(float3), cudaMemcpyDeviceToHost);

        for (unsigned int i=0;i<w;i++) {
            for (unsigned int j=0;j<h;j++) {                
                (*testOutputTexture)(i,j,0) = data[(i*h+j)*3+0]*255;
                (*testOutputTexture)(i,j,1) = data[(i*h+j)*3+1]*255;
                (*testOutputTexture)(i,j,2) = data[(i*h+j)*3+2]*255;
            }
        }
        testOutputTexture->RebindTexture();

        Descale(data,w,h);


        unsigned int index = idx;
        Vector<3,float> magnet(data[index*3], data[index*3+1], data[index*3+2]);
        logger.info << "reading index: " << index << " with value: " << magnet << logger.end;

        free(data);
    }
}

void MRIModule::Descale(float *data, int w, int h) {
    
    unsigned int nw=4, nh=4;

    unsigned int sx = w/nw;
    unsigned int sy = h/nh;

    for (unsigned int x=0; x<nw; x++)
        for (unsigned int y=0; y<nh; y++) {
            float sum_x = 0.0;
            float sum_y = 0.0;
            float sum_z = 0.0;

            for (unsigned int dx=0; dx<sx; dx++) 
                for (unsigned int dy=0; dy<sy; dy++) {
                    sum_x += data[((dx+sx)*h+(dy+sy))*3+0];
                    sum_y += data[((dx+sx)*h+(dy+sy))*3+1];
                    sum_z += data[((dx+sx)*h+(dy+sy))*3+2];
                }
            
            sum_x /= sx*sy;
            sum_y /= sx*sy;
            sum_z /= sx*sy;

            (*descaledOutputTexture)(x,y,0) = sum_x*255;
            (*descaledOutputTexture)(x,y,1) = sum_y*255;
            (*descaledOutputTexture)(x,y,3) = sum_z*255;
            descaledVectors[x][y] = Vector<3,float>(sum_x, sum_y, sum_z);
        }

    descaledOutputTexture->RebindTexture();
}

void MRIModule::Handle(InitializeEventArg arg) {
    INITIALIZE_CUDA();
    //logger.info << PRINT_CUDA_DEVICE_INFO() << logger.end;

    unsigned int w = img->GetWidth();
    unsigned int h = img->GetHeight();

    // logger.info << "max index: " << w*h << logger.end;

    float3* data = (float3*)malloc(sizeof(float3) * w * h);
    SpinProperty* ps = (SpinProperty*)malloc(sizeof(SpinProperty) * w * h);
    float scale = 1.0;

    UCharTexture2D* input = dynamic_cast<UCharTexture2D*>(img.get()); 

    for (unsigned int i=0; i < w;i++) {
        for (unsigned int j=0; j < h;j++) {
             Vector<4,unsigned char> pixel = input->GetPixelValues(i,j);
             float pix = (0.3*pixel[0] + 0.59*pixel[1] + 0.11*pixel[2]);
             pix /= 255;
        
             data[(i*h+j)] = make_float3(0.0, scale*pix, 0.0);
             // data[(i*h+j)*3+1] = scale*pix;
             // data[(i*h+j)*3+2] = 0.0;
             SpinProperty p;
             p.eq = scale*pix;
             p.t1 = (1e-5)*pix;
             p.t2 = (1e-6)*pix;
             ps[i*h+j] = p;
        }
    }

    cudaMalloc((void**)&lab_spins, sizeof(float3) * w * h);        
    cudaMalloc((void**)&ref_spins, sizeof(float3) * w * h);        
    cudaMalloc((void**)&props, sizeof(SpinProperty) * w * h);        
    cudaMemcpy(lab_spins, data, w * h * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(ref_spins, data, w * h * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(props, ps, w * h * sizeof(SpinProperty), cudaMemcpyHostToDevice);
    free(data);    
    free(ps);
}

void MRIModule::Handle(DeinitializeEventArg arg) {
    
}

void MRIModule::Handle(KeyboardEventArg arg) {
    if (arg.type != EVENT_RELEASE)
        return;
    if (arg.sym == KEY_k) {
        
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
    } else if (arg.sym == KEY_s) {
        // Step!
        
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
    unsigned int w = img->GetWidth();
    unsigned int h = img->GetHeight();

    MRI_INSPECTION(bool, Running, "running"); // simulation toggle
    MRI_INSPECTION(float, B0, "B0 (Tesla)");  // B0 field strength
    MRI_INSPECTION(bool, FID, "FID signal");  // B1 toggle
    MRI_INSPECTION(unsigned int, Index, "test index");  // 
    ((RWValue<unsigned int>*)values.back())->properties[MAX] = w*h;

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

