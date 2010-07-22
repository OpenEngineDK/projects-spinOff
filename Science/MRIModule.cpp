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
    , descaledOutputTexture(EmptyTextureResource::Create(DESCALE_W,DESCALE_H,24))
    , signalTexture(EmptyTextureResource::Create(SIGNAL_SIZE,100,24))
    , signalOutputTexture(EmptyTextureResource::Create(SIGNAL_SIZE,100,8))
    , signalOutput2Texture(EmptyTextureResource::Create(SIGNAL_SIZE,100,8))
    , plotTexture(EmptyTextureResource::Create(SIGNAL_SIZE,100,24))
    , plot(new Plot(Vector<2,float>(0,SIGNAL_SIZE), Vector<2,float>(-1,1)))
    , plotData1(new PointGraphDataSet(SIGNAL_SIZE, 0, SIGNAL_SIZE))
    , plotData2(new PointGraphDataSet(SIGNAL_SIZE, 0, SIGNAL_SIZE))
    , running(false)
    , fid(false)
    , sequence(false)
    , relax(0.0)
    , b0(.5)
    , gx(0.0)
    , gy(0.0)
    , fov(0.005)
    , phaseTime(1e-6)
    , lab_spins(NULL)
    , ref_spins(NULL)
    , props(NULL)
    , idx(99747)
    , theDT(1e-8)
    , theTime(0.0)
    , sigIdx(0,0)
    , signalData((cuFloatComplex*)malloc(sizeof(cuFloatComplex)*SIGNAL_SIZE*100))
 {
     phaseTime = theDT;
     plot->AddDataSet(plotData1);
     plotData2->SetColor(Vector<3,float>(0,0,1));
     plot->AddDataSet(plotData2);
 }

void MRIModule::Handle(Renderers::RenderingEventArg arg) {
    unsigned int w = DESCALE_W;
    unsigned int h = DESCALE_H;
    float size = 20.0;
    float space = 2.0;

    arg.renderer.ApplyViewingVolume(*arg.canvas.GetViewingVolume());

    for (unsigned int i=0;i<w;i++) {
        for (unsigned int j=0;j<h;j++) {                
            Vector<3,float> dir = descaledVectors[i][j]*size;
            arg.renderer.DrawLine(Line(Vector<3,float>(i,j,0.0)*space,
                                       Vector<3,float>(i*space+dir[0],
                                                       j*space+dir[1],
                                                       dir[2])), 
                                  Vector<3,float>(1.0,0.0,0.0), 2.0);
        }
    }
}

void MRIModule::FIDSequence() {
    unsigned int w = img->GetWidth();
    unsigned int h = img->GetHeight();
        
    float3 b = make_float3(0.0,0.0,b0);
    float dt = theDT;

    for (unsigned int i = 0; i < 4; i++) {

        if (relax > 0.0) {
            relax -= dt;
            theTime += dt;
            MRI_step(dt, theTime, (float3*)lab_spins, (float3*)ref_spins,
                     props, img->GetWidth(), img->GetHeight(), b, 0.0, 0.0);
            CHECK_FOR_CUDA_ERROR();
            cudaThreadSynchronize();
            return;
        }

        // initialize each line. flip 90 degree and apply phase gradient.
        if (sigIdx[0]  == 0) {
            theTime = 0.0;
            // the flip
            float3 _b = b + make_float3(Math::PI*0.5,0.0,0.0);
            MRI_step(0.0, theTime, (float3*)lab_spins, (float3*)ref_spins,
                     props, img->GetWidth(), img->GetHeight(), _b, 0.0, 0.0);
            cudaThreadSynchronize();
            // the phase
            float pt = phaseTime;
            gy = SIGNAL_SIZE / (2*GYROMAGNETIC_RATIO*fov*phaseTime);
            gy -= (2*gy/ SIGNAL_SIZE)*sigIdx[1];
            logger.info << "sigIdx[1] = " << sigIdx[1] << logger.end;
            logger.info << "gy = " << gy << logger.end;
            while (pt > 0) {
                pt -= dt;
                theTime += dt;
                MRI_step(dt, theTime, (float3*)lab_spins, (float3*)ref_spins,
                         props, img->GetWidth(), img->GetHeight(), b, 0.0, gy);
                CHECK_FOR_CUDA_ERROR();
                cudaThreadSynchronize();
            }
            gy = 0.0;
        }
        
        // relax with frequency gradient
        theTime += dt;
        float samplingRate = dt;
        gx = samplingRate / (GYROMAGNETIC_RATIO * fov);
        float3 signal = MRI_step(dt, theTime, (float3*)lab_spins, (float3*)ref_spins,
                                 props, w, h, b, gx, 0.0);
        
        //signal aquisition
        float signalScale = 0.1;
        if (sigIdx[1] < SIGNAL_SIZE) {
            // signal
            (*signalTexture)(sigIdx[0],sigIdx[1],0) = signalScale*signal.x*255;
            (*signalTexture)(sigIdx[0],sigIdx[1],1) = signalScale*signal.y*255;
            (*signalTexture)(sigIdx[0],sigIdx[1],2) = 0;//signal.z*255;
            
            plotData1->SetValue(sigIdx[0],
                                signal.x);
            plotData2->SetValue(sigIdx[0],
                                signal.y);


            signalData[sigIdx[1]*SIGNAL_SIZE+sigIdx[0]] 
                = make_cuFloatComplex(signal.x, signal.y);
            
            sigIdx[0]++;                
            if (sigIdx[0] == SIGNAL_SIZE) {
                sigIdx[0] = 0;
                sigIdx[1]++;                    
                
                relax = 1e-6;
            }
        }
    }
    plot->RenderInEmptyTexture(plotTexture);
    plotTexture->RebindTexture();

    signalTexture->RebindTexture();
}

void MRIModule::Handle(ProcessEventArg arg) {
    if (!running) return;
    
    unsigned int w = img->GetWidth();
    unsigned int h = img->GetHeight();

    if (sequence) {
        FIDSequence();
    }
    else {
        float3 b = make_float3(0.0,0.0,b0);
        if (fid) {
            logger.info << "FID" << logger.end;
            float3 _b = b + make_float3(Math::PI*0.5,0.0,0.0);
            float3 signal = MRI_step(0.0, theTime, (float3*)lab_spins, (float3*)ref_spins,
                                     props, w, h, _b, 0.0, 0.0);
            fid = false;
            cudaThreadSynchronize();
        }
        float dt = theDT;
        theTime += dt;
        float3 signal = MRI_step(dt, theTime, (float3*)lab_spins, (float3*)ref_spins,
                                 props, img->GetWidth(), img->GetHeight(), b, gx, gy);
        
    }

    // copy result to textures 'n stuff
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
    free(data);
}

void MRIModule::Descale(float *data, int w, int h) {
    
    unsigned int
        nw=DESCALE_W,
        nh=DESCALE_H;

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
            (*descaledOutputTexture)(x,y,2) = sum_z*255;
            
            
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
        
             data[(i*h+j)] = make_float3(0.0, 0.0, scale*pix);

             // data[(i*h+j)*3+1] = scale*pix;
             // data[(i*h+j)*3+2] = 0.0;
             SpinProperty p;
             p.eq = scale*pix;
             p.t1 = (1e-5)*pix+(1e-6);
             p.t2 = (1e-6)*pix+(1e-7);
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
        cudaThreadSynchronize();

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
        cudaThreadSynchronize();

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
    } else if (arg.sym == KEY_m) {
        logger.info << "Make signal output!" << logger.end;
        unsigned int w = SIGNAL_SIZE;
        unsigned int h = 100;
        // make the output image!
        cuFloatComplex* data = (cuFloatComplex*)malloc(sizeof(cuFloatComplex) * w * h);

        cuFloatComplex *devData;
        cudaMalloc((void**)&devData, sizeof(cuFloatComplex)*w*h);
        cudaMemcpy(devData, signalData, sizeof(cuFloatComplex)*w*h, cudaMemcpyHostToDevice);
        K2I_ALL(devData, make_uint2(w,h));
        
        cudaMemcpy(data, devData, w * h * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

        float maxl = 0;
        for (unsigned int i=0;i<w;i++) {
            for (unsigned int j=0;j<h;j++) {
                cuFloatComplex c = data[i*h+j];
                float l = sqrt(c.x*c.x + c.y*c.y);
                //logger.info << l << logger.end;
                maxl = (fmax(maxl,l));
            }
        }

        logger.info << maxl << logger.end;

        for (unsigned int i=0;i<w;i++) {
            for (unsigned int j=0;j<h;j++) {
                cuFloatComplex c = data[i*h+j];
                (*signalOutputTexture)(i,j) = (255/maxl)*sqrt(c.x*c.x + c.y*c.y);

            }
        }
        signalOutputTexture->RebindTexture();

        cudaMemcpy(devData, signalData, sizeof(cuFloatComplex)*w*h, cudaMemcpyHostToDevice);
        I2K_ALL(devData, make_uint2(w,h));
        
        cudaMemcpy(data, devData, w * h * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

        maxl = 0;
        for (unsigned int i=0;i<w;i++) {
            for (unsigned int j=0;j<h;j++) {
                cuFloatComplex c = data[i*h+j];
                float l = sqrt(c.x*c.x + c.y*c.y);
                //logger.info << l << logger.end;
                maxl = (fmax(maxl,l));
            }
        }

        logger.info << maxl << logger.end;

        for (unsigned int i=0;i<w;i++) {
            for (unsigned int j=0;j<h;j++) {
                cuFloatComplex c = data[i*h+j];
                (*signalOutput2Texture)(i,j) = (255/maxl)*sqrt(c.x*c.x + c.y*c.y);

            }
        }
        signalOutput2Texture->RebindTexture();

        

        cudaFree(devData);
        free(data);
    }
}

using namespace Utils::Inspection;

#define MRI_INSPECTION(type, field, _name)                              \
    INSPECT_VALUE(MRIModule, type, field, _name)

ValueList MRIModule::Inspection() {
    ValueList values;
    unsigned int w = img->GetWidth();
    unsigned int h = img->GetHeight();

    MRI_INSPECTION(bool, Running, "running"); // simulation toggle
    MRI_INSPECTION(bool, Sequence, "Run FID Sequence"); // start recording
    MRI_INSPECTION(float, B0, "B0 (Tesla)");  // B0 field strength
    MRI_INSPECTION(float, FOV, "Field of view");  // Field of view
    MRI_INSPECTION(float, PhaseTime, "Phase time (s)");  // Field of view
    MRI_INSPECTION(float, Gx, "Gradient X (Tesla)");  // Gradient x component field strength
    MRI_INSPECTION(float, Gy, "Gradient y (Tesla)");  // Gradient y component field strength
    MRI_INSPECTION(bool, FID, "FID signal");  // B1 toggle
    {
        MRI_INSPECTION(unsigned int, Index, "test index");
        ((RWValue<unsigned int>*)values.back())->properties[MAX] = w*h;
    }
    {
        MRI_INSPECTION(float, DT, "dt (sec)");  // Delta time
        ((RWValue<float>*)values.back())->properties[MIN] = 1e-9;
        ((RWValue<float>*)values.back())->properties[MAX] = 1e-6;
        ((RWValue<float>*)values.back())->properties[STEP] = 1e-9;
 
    }
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

