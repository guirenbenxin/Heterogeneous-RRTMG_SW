#include <iostream>

#include <stdio.h>
#include <cuda_runtime.h>
#include<cmath>
#include<stdlib.h>
#include <time.h>

#include <thrust/extrema.h>

#define ncol 2048
#define nlay 51
#define nlayers 52
#define nbndsw 14
#define ngptsw 112
#define jpb1 16
#define jpb2 29
#define jpband 29
#define mxmol 38
#define nmol 7
#define tpb 8
#define od_lo 0.06
#define tblint 2
#define pi 2.0*asin(1.0)
#define min(x,y)(((x)<(y))?(x):(y))
#define max(x,y)(((x)>(y))?(x):(y))

__device__ int inflag;
__device__ int iceflag;
__device__ int liqflag;

/////////////////////////////////////////////////////////////////////
//子程序inatm从GCM读取大气轮廓线，以用于RRTMG_SW，并定义其他输入参数
/////////////////////////////////////////////////////////////////////
//二维加速
__global__ void inatm_d1(double *wkl,
						double *reicmc,double *dgesmc,
						double *relqmc,double *taua,double *ssaa,double *asma)
{
    int iplon,lay;
    int i;
	iplon=blockDim.x * blockIdx.x + threadIdx.x;
	lay=blockDim.y * blockIdx.y + threadIdx.y;

	if(((iplon>=0)&&(iplon<ncol))&&((lay>=0)&&(lay<nlayers)))
    {

        for(i=0;i<38;i++)
            wkl[lay*mxmol*ncol+i*ncol+iplon]=0.0;

				reicmc[lay*ncol+iplon]=0.0;
				dgesmc[lay*ncol+iplon]=0.0;
				relqmc[lay*ncol+iplon]=0.0;
                                                  
			for(i=0;i<nbndsw;i++)
			{
				taua[i*nlayers*ncol+lay*ncol+iplon]=0.0;
				asma[i*nlayers*ncol+lay*ncol+iplon]=0.0;
				ssaa[i*nlayers*ncol+lay*ncol+iplon]=1.0;
			}
    }
    return;
}

//二维加速
__global__ void inatm_d2(double *play_d,double *plev_d,double *tlay_d,double *tlev_d,
						double *h2ovmr_d,double *o3vmr_d,double *co2vmr_d,double *ch4vmr_d,
						double *o2vmr_d,double *n2ovmr_d,
						double *pavel,double *pz,double *pdp,
						double *tavel,double *tz,double *wkl)
{
    int iplon,lay;
    iplon=blockDim.x * blockIdx.x + threadIdx.x;
    lay=blockDim.y * blockIdx.y + threadIdx.y;

	if(((iplon>=0)&&(iplon<ncol))&&((lay>=0)&&(lay<nlayers-1)))
    {

			pz[0*ncol+iplon]=plev_d[(nlayers-1)*ncol+iplon];
			tz[0*ncol+iplon]=tlev_d[(nlayers-1)*ncol+iplon];
        	pavel[lay*ncol+iplon]=play_d[(nlayers-lay-2)*ncol+iplon];
			tavel[lay*ncol+iplon]=tlay_d[(nlayers-lay-2)*ncol+iplon];
			pz[(lay+1)*ncol+iplon]=plev_d[(nlayers-lay-2)*ncol+iplon];
			tz[(lay+1)*ncol+iplon]=tlev_d[(nlayers-lay-2)*ncol+iplon];
			pdp[lay*ncol+iplon]=pz[lay*ncol+iplon]-pz[(lay+1)*ncol+iplon];
		// For h2o input in vmr:
			wkl[lay*mxmol*ncol+0*ncol+iplon]=h2ovmr_d[(nlayers-lay-2)*ncol+iplon];
			wkl[lay*mxmol*ncol+1*ncol+iplon]=co2vmr_d[(nlayers-lay-2)*ncol+iplon];
			wkl[lay*mxmol*ncol+2*ncol+iplon]=o3vmr_d[(nlayers-lay-2)*ncol+iplon];
			wkl[lay*mxmol*ncol+3*ncol+iplon]=n2ovmr_d[(nlayers-lay-2)*ncol+iplon];
			wkl[lay*mxmol*ncol+5*ncol+iplon]=ch4vmr_d[(nlayers-lay-2)*ncol+iplon];
			wkl[lay*mxmol*ncol+6*ncol+iplon]=o2vmr_d[(nlayers-lay-2)*ncol+iplon];
		
    }

    return;
}
//三维加速
__global__ void inatm_d3(int icld,double *cldfmcl_d,double *taucmcl_d,
						double *ssacmcl_d,double *asmcmcl_d,double *fsfcmcl_d,double *ciwpmcl_d,
						double *clwpmcl_d,double *reicmcl_d,double *relqmcl_d,double *tauaer_d,
						double *cldfmc,double *taucmc,double *ssacmc,double *asmcmc,
						double *fsfcmc,double *ciwpmc,double *clwpmc)
{
    int iplon,lay,ig;
    ig=blockDim.x * blockIdx.x + threadIdx.x;
	iplon=blockDim.y * blockIdx.y + threadIdx.y;
	lay=blockDim.z * blockIdx.z + threadIdx.z;

	//在数组的第三维最后一组数据在inatm_d4中分配
	if(((iplon>=0)&&(iplon<ncol))&&((ig>=0)&&(ig<ngptsw))&&((lay>=0)&&(lay<nlayers-1)))
    {

        if(icld>=1)
        {
            //为3D数组分配值
            cldfmc[lay*ngptsw*ncol+ig*ncol+iplon]=cldfmcl_d[(nlay-lay-1)*ngptsw*ncol+ig*ncol+iplon];
            taucmc[lay*ngptsw*ncol+ig*ncol+iplon]=taucmcl_d[(nlay-lay-1)*ngptsw*ncol+ig*ncol+iplon];
            ssacmc[lay*ngptsw*ncol+ig*ncol+iplon]=ssacmcl_d[(nlay-lay-1)*ngptsw*ncol+ig*ncol+iplon];
            asmcmc[lay*ngptsw*ncol+ig*ncol+iplon]=asmcmcl_d[(nlay-lay-1)*ngptsw*ncol+ig*ncol+iplon];
            fsfcmc[lay*ngptsw*ncol+ig*ncol+iplon]=fsfcmcl_d[(nlay-lay-1)*ngptsw*ncol+ig*ncol+iplon];
            ciwpmc[lay*ngptsw*ncol+ig*ncol+iplon]=ciwpmcl_d[(nlay-lay-1)*ngptsw*ncol+ig*ncol+iplon];
            clwpmc[lay*ngptsw*ncol+ig*ncol+iplon]=clwpmcl_d[(nlay-lay-1)*ngptsw*ncol+ig*ncol+iplon];
        }
    }
    return;
}

//二维加速
__global__ void inatm_d4(int icld,int iaer,int inflgsw,int iceflgsw,int liqflgsw,
					    double *reicmcl_d,double *relqmcl_d,double *tauaer_d,
						double *ssaaer_d,double *asmaer_d,
						double *reicmc,double *dgesmc,
						double *relqmc,double *taua,double *ssaa,double *asma)
{
    int iplon,lay;
    int ib;   //Loop indices循环索引
    int inflag;
	int iceflag;
	int liqflag;
	iplon=blockDim.x * blockIdx.x + threadIdx.x;
	lay=blockDim.y * blockIdx.y + threadIdx.y;

	if(((iplon>=0)&&(iplon<ncol))&&((lay>=0)&&(lay<nlayers-1)))
    {
        if(iaer>=1)
		{
				for(ib=0;ib<nbndsw;ib++)
				{
					taua[ib*nlayers*ncol+lay*ncol+iplon]=tauaer_d[ib*nlay*ncol+(nlay-lay-1)*ncol+iplon];
					ssaa[ib*nlayers*ncol+lay*ncol+iplon]=ssaaer_d[ib*nlay*ncol+(nlay-lay-1)*ncol+iplon];
					asma[ib*nlayers*ncol+lay*ncol+iplon]=asmaer_d[ib*nlay*ncol+(nlay-lay-1)*ncol+iplon];
				}

		}

		if(icld>=1)
        {
            inflag=inflgsw;
			iceflag=iceflgsw;
			liqflag=liqflgsw;

			reicmc[lay*ncol+iplon]=reicmcl_d[(nlay-lay-1)*ncol+iplon];
				if(iceflag==3)
				{
					dgesmc[lay*ncol+iplon]=1.5396*reicmcl_d[(nlay-lay-1)*ncol+iplon];
				}
				relqmc[lay*ncol+iplon]=relqmcl_d[(nlay-lay-1)*ncol+iplon];
        }
    }
    return;
}

//仍使用一维加速
__global__ void inatm_d5(double avogad,double grav,
                        double adjes,int dyofyr,
						double *tsfc_d,
						double *solvar_d,
						double *pavel,double *pz,double *pdp,
						double *tavel,double *tz,double *tbound,double *adjflux,double *wkl,
						double *coldry,double *cldfmc,double *taucmc,double *ssacmc,double *asmcmc,
						double *fsfcmc,double *ciwpmc,double *clwpmc,double *reicmc,double *dgesmc,
						double *relqmc,double *taua,double *ssaa,double *asma
						)
{
    //----- Local -----
	int iplon,lay;

	int imol,ib;   //Loop indices循环索引
	double amm,adjflx;
	//real(kind=r8) :: gamma
	const double amd=28.9660;        //干燥空气的有效分子量（g / mol）
	const double amw=18.0160;        //水蒸气的分子量（g / mol）
	const double amdw=1.607793;      //干燥空气/水蒸气的分子量
	const double amdc=0.658114;      //干燥空气/二氧化碳的分子量
	const double amdo= 0.603428;     //干燥空气/臭氧的分子量
	const double amdm= 1.805423;     //干燥空气/甲烷的分子量
	const double amdn= 0.658090;     //干燥空气/一氧化二氮的分子量
	const double amdc1=0.210852;     //干空气的分子量/ CFC11
	const double amdc2 = 0.239546;   //干空气/ CFC12的分子量
	const double sbc = 5.67e-08 ;    //Stefan-Boltzmann常数（W / m2K4）

	iplon=blockDim.x * blockIdx.x + threadIdx.x;

	if((iplon>=0)&&(iplon<ncol))
    {

        adjflx = adjes;
		if(dyofyr > 0)
		{
		 adjflx = 1.000110 + 0.034221 * cos(2.0*pi*(dyofyr-1.0)/365.0) + 0.001289 * sin(2.0*pi*(dyofyr-1)/365.0) +
                  0.000719 * cos(2.0*pi*(dyofyr-1)/365.0) + 0.000077 * sin(2.0*pi*(dyofyr-1.0)/365.0);
		}
		for(ib=0;ib<jpband;ib++)
		{
			adjflux[ib*ncol+iplon]=0.0;
		}
		for(ib=jpb1-1;ib<jpb2;ib++)
		{
			adjflux[ib*ncol+iplon]= adjflx * solvar_d[ib*ncol+iplon];
		}

		tbound[iplon]=tsfc_d[iplon];

		for(lay=0;lay<nlay;lay++)
        {
            amm=((double(1.0)-wkl[lay*mxmol*ncol+0*ncol+iplon])*amd)+(wkl[lay*mxmol*ncol+0*ncol+iplon]*amw);
			coldry[lay*ncol+iplon]=(pz[lay*ncol+iplon]-pz[(lay+1)*ncol+iplon])*double(1.e3)*avogad/(double(1.e2)*grav*amm*(double(1.0)+wkl[lay*mxmol*ncol+0*ncol+iplon]));
        }

        pavel[nlay*ncol+iplon]=double(0.5)*pz[(nlayers-1)*ncol+iplon];
		tavel[nlay*ncol+iplon]=tavel[(nlay-1)*ncol+iplon] ;
		pz[nlayers*ncol+iplon]=double(1.e-4);
		tz[nlay*ncol+iplon]=double(0.5)*(tavel[nlay*ncol+iplon]+tavel[(nlay-1)*ncol+iplon]);
		tz[nlayers*ncol+iplon]=tz[nlay*ncol+iplon];
		pdp[nlay*ncol+iplon]=pz[(nlayers-1)*ncol+iplon]-pz[nlayers*ncol+iplon];
		wkl[nlay*mxmol*ncol+0*ncol+iplon]=wkl[(nlay-1)*mxmol*ncol+0*ncol+iplon];
		wkl[nlay*mxmol*ncol+1*ncol+iplon]=wkl[(nlay-1)*mxmol*ncol+1*ncol+iplon];
		wkl[nlay*mxmol*ncol+2*ncol+iplon]=wkl[(nlay-1)*mxmol*ncol+2*ncol+iplon];
		wkl[nlay*mxmol*ncol+3*ncol+iplon]=wkl[(nlay-1)*mxmol*ncol+3*ncol+iplon];
		wkl[nlay*mxmol*ncol+5*ncol+iplon]=wkl[(nlay-1)*mxmol*ncol+5*ncol+iplon];
		wkl[nlay*mxmol*ncol+6*ncol+iplon]=wkl[(nlay-1)*mxmol*ncol+6*ncol+iplon];
		amm=(double(1.0)- wkl[(nlay-1)*mxmol*ncol+0*ncol+iplon])*amd +wkl[(nlay-1)*mxmol*ncol+0*ncol+iplon]*amw;
		coldry[nlay*ncol+iplon]= pz[(nlayers-1)*ncol+iplon] *double(1.e3)*avogad/(double(1.e2)*grav*amm*(double(1.0)+wkl[(nlay-1)*mxmol*ncol+0*ncol+iplon])) ;

        for(lay=0;lay<nlayers;lay++)
		{
			for(imol=0;imol<nmol;imol++)
			{
				wkl[lay*mxmol*ncol+imol*ncol+iplon]=coldry[lay*ncol+iplon]*wkl[lay*mxmol*ncol+imol*ncol+iplon];
			}
		}
		/*****************************************************/
		// If an extra layer is being used in RRTMG, set all cloud properties to zero in the extra layer.
		//如果RRTMG中使用了额外的层，请将额外层中的所有云属性设置为零
		for(int i=0;i<ngptsw;i++)
		{
			cldfmc[(nlayers-1)*ngptsw*ncol+i*ncol+iplon]=0.0;
			taucmc[(nlayers-1)*ngptsw*ncol+i*ncol+iplon]=0.0;
			ssacmc[(nlayers-1)*ngptsw*ncol+i*ncol+iplon]=1.0;
			asmcmc[(nlayers-1)*ngptsw*ncol+i*ncol+iplon]=0.0;
			fsfcmc[(nlayers-1)*ngptsw*ncol+i*ncol+iplon]=0.0;
			ciwpmc[(nlayers-1)*ngptsw*ncol+i*ncol+iplon]=0.0;
			clwpmc[(nlayers-1)*ngptsw*ncol+i*ncol+iplon]=0.0;
		}
		reicmc[(nlayers-1)*ncol+iplon]=0.0;
		dgesmc[(nlayers-1)*ncol+iplon]=0.0;
		relqmc[(nlayers-1)*ncol+iplon]=0.0;
		for(int i=0;i<nbndsw;i++)
		{
			taua[i*nlayers*ncol+(nlayers-1)*ncol+iplon]=0.0;
			ssaa[i*nlayers*ncol+(nlayers-1)*ncol+iplon]=1.0;
			asma[i*nlayers*ncol+(nlayers-1)*ncol+iplon]=0.0;
		}
    }
    return;
}

///////////////////////////////////////////////////////////////////////
//调用子程序cldprmc为集云光学深度MclCA基于输入云性能
///////////////////////////////////////////////////////////////////////

__global__ void cldprmc_d(double *taormc,double *taucmc,double *ciwpmc,double *clwpmc,
						double *cldfmc,double *fsfcmc,double *ssacmc,double *asmcmc,
						double *reicmc,double *wavenum2,double *abari,double *bbari,
						double *cbari,double *dbari,double *ebari,double *fbari,
						double *ngb,double *extice2,double *ssaice2,double *asyice2,
						double *dgesmc,double *extice3,double *ssaice3,double *asyice3,
						double *fdlice3,double *relqmc,double *extliq1,double *ssaliq1,
						double *asyliq1)
{
	// ------- Local -------
	int iplon;
	int ib,lay,istr,index1,icx,ig;

	const double eps=1.e-6;
	const double cldmin=1.e-80;
	double cwp,radliq,radice,dgeice,factor,fint;

	double taucldorig_a, taucloud_a, ssacloud_a, ffp, ffp1, ffpssa;
	double tauiceorig, scatice, ssaice, tauice, tauliqorig, scatliq, ssaliq, tauliq;

	double fdelta[112];
	double extcoice[112],gice[112];
	double ssacoice[112],forwice[112];
	double extcoliq[112],gliq[112];
	double ssacoliq[112],forwliq[112];

	ig=blockDim.x * blockIdx.x + threadIdx.x;
	iplon=blockDim.y * blockIdx.y + threadIdx.y;
	lay=blockDim.z * blockIdx.z + threadIdx.z;
	//iplon=blockDim.x * blockIdx.x + threadIdx.x;
    //lay=blockDim.y * blockIdx.y + threadIdx.y;
    if(((iplon>=0)&&(iplon<ncol))&&((ig>=0)&&(ig<ngptsw))&&((lay>=0)&&(lay<nlayers)))
	//if((iplon>=0)&&(iplon<ncol)&&((lay>=0)&&(lay<nlayers)))
	{

				taormc[lay*ngptsw*ncol+ig*ncol+iplon]=taucmc[lay*ngptsw*ncol+ig*ncol+iplon];

				cwp=ciwpmc[lay*ngptsw*ncol+ig*ncol+iplon]+clwpmc[lay*ngptsw*ncol+ig*ncol+iplon];
				if((cldfmc[lay*ngptsw*ncol+ig*ncol+iplon]>=cldmin)&& (cwp>=cldmin||taucmc[lay*ngptsw*ncol+ig*ncol+iplon]>=cldmin))
				{
					//(inflag=0): Cloud optical properties input directly
					//Cloud optical properties already defined in taucmc, ssacmc, asmcmc are unscaled;
                    // Apply delta-M scaling here (using Henyey-Greenstein approximation)
					if(inflag==0)
					{
						taucldorig_a = taucmc[lay*ngptsw*ncol+ig*ncol+iplon];
						ffp=fsfcmc[lay*ngptsw*ncol+ig*ncol+iplon];
						ffp1=(double)1.0-ffp;
						ffpssa=(double)1.0-ffp*ssacmc[lay*ngptsw*ncol+ig*ncol+iplon];
						ssacloud_a = ffp1 * ssacmc[lay*ngptsw*ncol+ig*ncol+iplon] / ffpssa;
						taucloud_a = ffpssa * taucldorig_a;

						taormc[lay*ngptsw*ncol+ig*ncol+iplon] = taucldorig_a;
						ssacmc[lay*ngptsw*ncol+ig*ncol+iplon] = ssacloud_a;
						taucmc[lay*ngptsw*ncol+ig*ncol+iplon] = taucloud_a;
						asmcmc[lay*ngptsw*ncol+ig*ncol+iplon] = (asmcmc[lay*ngptsw*ncol+ig*ncol+iplon] - ffp) / (ffp1);
					}

					//elseif (inflag .eq. 1) then
                 //stop 'INFLAG = 1 OPTION NOT AVAILABLE WITH MCICA'

				 //(inflag=2): Separate treatement of ice clouds and water clouds.
				    if(inflag==2)
					{
						radice = reicmc[lay*ncol+iplon];
						//Calculation of absorption coefficients due to ice clouds.
						if(ciwpmc[lay*ngptsw*ncol+ig*ncol+iplon]==0.0)
						{
							extcoice[ig] = 0.0;
							ssacoice[ig] = 0.0;
							gice[ig] = 0.0;
							forwice[ig] = 0.0;
						}
						//(iceflag = 1):
						//Note: This option uses Ebert and Curry approach for all particle sizes similar to
						//CAM3 implementation, though this is somewhat unjustified for large ice particles
						else if(iceflag==1)
						{
							ib=ngb[ig];
							if(wavenum2[ib-16]>1.43e4)
								icx=0;
							else if(wavenum2[ib-16]>7.7e3)
								icx=1;
							else if(wavenum2[ib-16]>5.3e3)
								icx=2;
							else if(wavenum2[ib-16]>4.0e3)
								icx=3;
							else if(wavenum2[ib-16]>2.5e3)
								icx=4;

							extcoice[ig] = (abari[icx] + bbari[icx]/radice);
							ssacoice[ig] = double(1.0)- cbari[icx] - dbari[icx] * radice;
							gice[ig] = ebari[icx] + fbari[icx] * radice;
						//Check to ensure upper limit of gice is within physical limits for large particles
							if(gice[ig]>=1.0)
								gice[ig]=1.0-eps;
							forwice[ig]=gice[ig]*gice[ig];
/*Check to ensure all calculated quantities are within physical limits.
                     !if (extcoice(ig) .lt. 0.0_r8) stop 'ICE EXTINCTION LESS THAN 0.0'
                    ! if (ssacoice(ig) .gt. 1.0_r8) stop 'ICE SSA GRTR THAN 1.0'
                    ! if (ssacoice(ig) .lt. 0.0_r8) stop 'ICE SSA LESS THAN 0.0'
                    ! if (gice(ig) .gt. 1.0_r8) stop 'ICE ASYM GRTR THAN 1.0'
                     !if (gice(ig) .lt. 0.0_r8) stop 'ICE ASYM LESS THAN 0.0'

! For iceflag=2 option, combine with iceflag=0 option to handle large particle sizes.
! Use iceflag=2 option for ice particle effective radii from 5.0 to 131.0 microns
! and use iceflag=0 option for ice particles greater than 131.0 microns.
! *** NOTE: Transition between two methods has not been smoothed.
*/
						}
						else if(iceflag==2)
						{
							//if (radice .lt. 5.0_r8) stop 'ICE RADIUS OUT OF BOUNDS'
							if((radice>=5.0)&&(radice<=131.0))
							{
								factor=(radice-2.0)/3.0;
								index1=int(factor);
								if(index1==43)
									index1=42;
								fint=factor-float(index1);
								ib=ngb[ig];
								extcoice[ig] = extice2[(ib-16)*43+(index1-1)] + fint *
                                      (extice2[(ib-16)*43+(index1-1)]-extice2[(ib-16)*43+(index1-1)]);
								ssacoice[ig] = ssaice2[(ib-16)*43+(index1-1)] + fint *
                                      (ssaice2[(ib-16)*43+(index1-1)]-ssaice2[(ib-16)*43+(index1-1)]);
								gice[ig] = asyice2[(ib-16)*43+(index1-1)] + fint *
                                      (asyice2[(ib-16)*43+(index1-1)]-asyice2[(ib-16)*43+(index1-1)]);
								forwice[ig] = gice[ig]*gice[ig];

							/*	! Check to ensure all calculated quantities are within physical limits.
								! if (extcoice(ig) .lt. 0.0_r8) stop 'ICE EXTINCTION LESS THAN 0.0'
								! if (ssacoice(ig) .gt. 1.0_r8) stop 'ICE SSA GRTR THAN 1.0'
								! if (ssacoice(ig) .lt. 0.0_r8) stop 'ICE SSA LESS THAN 0.0'
								!if (gice(ig) .gt. 1.0_r8) stop 'ICE ASYM GRTR THAN 1.0'
								!if (gice(ig) .lt. 0.0_r8) stop 'ICE ASYM LESS THAN 0.0'*/

							}
							else if(radice>131.0)
							{
								ib=ngb[ig];
								if(wavenum2[ib-16]>1.43e4)
									icx=0;
								else if(wavenum2[ib-16]>7.7e3)
									icx=1;
								else if(wavenum2[ib-16]>5.3e3)
									icx=2;
								else if(wavenum2[ib-16]>4.0e3)
									icx=3;
								else if(wavenum2[ib-16]>2.5e3)
									icx=4;

								extcoice[ig] = (abari[icx] + bbari[icx]/radice);
								ssacoice[ig] = double(1.0)- cbari[icx] - dbari[icx] * radice;
								gice[ig] = ebari[icx] + fbari[icx] * radice;
						//Check to ensure upper limit of gice is within physical limits for large particles
								if(gice[ig]>=1.0)
									gice[ig]=1.0-eps;
								forwice[ig]=gice[ig]*gice[ig];
								/*! Check to ensure all calculated quantities are within physical limits.
                        !if (extcoice(ig) .lt. 0.0_r8) stop 'ICE EXTINCTION LESS THAN 0.0'
                        !if (ssacoice(ig) .gt. 1.0_r8) stop 'ICE SSA GRTR THAN 1.0'
                       ! if (ssacoice(ig) .lt. 0.0_r8) stop 'ICE SSA LESS THAN 0.0'
                       ! if (gice(ig) .gt. 1.0_r8) stop 'ICE ASYM GRTR THAN 1.0'
                        !if (gice(ig) .lt. 0.0_r8) stop 'ICE ASYM LESS THAN 0.0'
						*/

							}
						}
/*For iceflag=3 option, combine with iceflag=0 option to handle large particle sizes
! Use iceflag=3 option for ice particle effective radii from 3.2 to 91.0 microns
! (generalized effective size, dge, from 5 to 140 microns), and use iceflag=0 option
! for ice particle effective radii greater than 91.0 microns (dge = 140 microns).
! *** NOTE: Fu parameterization requires particle size in generalized effective size.
! *** NOTE: Transition between two methods has not been smoothed.
*/
						else if(iceflag==3)
						{
							dgeice=dgesmc[lay*ncol+iplon];
							//if (dgeice .lt. 5.0_r8) stop 'ICE GENERALIZED EFFECTIVE SIZE OUT OF BOUNDS'
							if((dgeice>=5.0) && (dgeice<=140.0))
							{
								factor=(dgeice-2.0)/3.0;
								index1=int(factor);
								if(index1==46)
									index1=45;
								fint=factor-float(index1);
								ib=ngb[ig];
								extcoice[ig] = extice3[(ib-16)*46+(index1-1)] + fint *
                                      (extice3[(ib-16)*46+index1]-extice3[(ib-16)*46+(index1-1)]);
								ssacoice[ig] = ssaice3[(ib-16)*46+(index1-1)] + fint *
                                      (ssaice3[(ib-16)*46+index1]-ssaice3[(ib-16)*46+(index1-1)]);
								gice[ig] = asyice3[(ib-16)*46+(index1-1)] + fint *
                                      (asyice3[(ib-16)*46+index1]-asyice3[(ib-16)*46+(index1-1)]);
								fdelta[ig]=fdlice3[(ib-16)*46+(index1-1)] + fint *
								      (fdlice3[(ib-16)*46+index1]-fdlice3[(ib-16)*46+(index1-1)]);
								//if (fdelta(ig) .lt. 0.0_r8) stop 'FDELTA LESS THAN 0.0'
								//if (fdelta(ig) .gt. 1.0_r8) stop 'FDELTA GT THAN 1.0'
								forwice[ig]=fdelta[ig] + 0.5/ssacoice[ig];
					//See Fu 1996 p. 2067
								if (forwice[ig] > gice[ig])
									forwice[ig] = gice[ig];
					/*! Check to ensure all calculated quantities are within physical limits.
                        !if (extcoice(ig) .lt. 0.0_r8) stop 'ICE EXTINCTION LESS THAN 0.0'
                        !if (ssacoice(ig) .gt. 1.0_r8) stop 'ICE SSA GRTR THAN 1.0'
                        !if (ssacoice(ig) .lt. 0.0_r8) stop 'ICE SSA LESS THAN 0.0'
                        !if (gice(ig) .gt. 1.0_r8) stop 'ICE ASYM GRTR THAN 1.0'
                        !if (gice(ig) .lt. 0.0_r8) stop 'ICE ASYM LESS THAN 0.0'
                    */

							}
							else if(dgeice > 140.0)
							{
								ib=ngb[ig];
								if(wavenum2[ib-16]>1.43e4)
									icx=0;
								else if(wavenum2[ib-16]>7.7e3)
									icx=1;
								else if(wavenum2[ib-16]>5.3e3)
									icx=2;
								else if(wavenum2[ib-16]>4.0e3)
									icx=3;
								else if(wavenum2[ib-16]>2.5e3)
									icx=4;

								extcoice[ig] = (abari[icx] + bbari[icx]/radice);
								ssacoice[ig] = double(1.0)- cbari[icx] - dbari[icx] * radice;
								gice[ig] = ebari[icx] + fbari[icx] * radice;
						//! Check to ensure upper limit of gice is within physical limits for large particles
								if(gice[ig]>=1.0)
									gice[ig]=1.0-eps;
								forwice[ig]=gice[ig]*gice[ig];
						/*Check to ensure all calculated quantities are within physical limits.
                        !if (extcoice(ig) .lt. 0.0_r8) stop 'ICE EXTINCTION LESS THAN 0.0'
                        !if (ssacoice(ig) .gt. 1.0_r8) stop 'ICE SSA GRTR THAN 1.0'
                        !if (ssacoice(ig) .lt. 0.0_r8) stop 'ICE SSA LESS THAN 0.0'
                        !if (gice(ig) .gt. 1.0_r8) stop 'ICE ASYM GRTR THAN 1.0'
                        !if (gice(ig) .lt. 0.0_r8) stop 'ICE ASYM LESS THAN 0.0'
						*/
							}

						}


				//Calculation of absorption coefficients due to water clouds.
						if(clwpmc[lay*ngptsw*ncol+ig*ncol+iplon]==0.0)
						{
							extcoliq[ig] = 0.0;
							ssacoliq[ig] = 0.0;
							gliq[ig] = 0.0;
							forwliq[ig] = 0.0;
						}
						else if(liqflag==1)
						{
							radliq = relqmc[lay*ncol+iplon];
							//if (radliq .lt. 1.5_r8 .or. radliq .gt. 60._r8) stop &
							//'liquid effective radius out of bounds'
							index1 = int(radliq - 1.5);
							if (index1==0)
								index1 = 1;
							if (index1==58)
								index1 = 57;
							fint = radliq - 1.5 - float(index1);
							ib = ngb[ig];
							extcoliq[ig] = extliq1[(ib-16)*58+index1-1] + fint *
                                   (extliq1[(ib-16)*58+index1] - extliq1[(ib-16)*58+index1-1]);
							ssacoliq[ig] = ssaliq1[(ib-16)*58+index1-1] + fint *
                                   (ssaliq1[(ib-16)*58+index1-1] - ssaliq1[(ib-16)*58+index1-1]);
							if ((fint<0) && (ssacoliq[ig]>1.0))
								ssacoliq[ig] = ssaliq1[(ib-16)*58+index1-1];
							gliq[ig] = asyliq1[(ib-16)*58+index1-1] + fint *
								(asyliq1[(ib-16)*58+index1] - asyliq1[(ib-16)*58+index1-1]);
							forwliq[ig] = gliq[ig]*gliq[ig];
							/*! Check to ensure all calculated quantities are within physical limits.
							! if (extcoliq(ig) .lt. 0.0_r8) stop 'LIQUID EXTINCTION LESS THAN 0.0'
							! if (ssacoliq(ig) .gt. 1.0_r8) stop 'LIQUID SSA GRTR THAN 1.0'
							! if (ssacoliq(ig) .lt. 0.0_r8) stop 'LIQUID SSA LESS THAN 0.0'
							! if (gliq(ig) .gt. 1.0_r8) stop 'LIQUID ASYM GRTR THAN 1.0'
							! if (gliq(ig) .lt. 0.0_r8) stop 'LIQUID ASYM LESS THAN 0.0'
							*/
						}
						tauliqorig = clwpmc[lay*ngptsw*ncol+ig*ncol+iplon] * extcoliq[ig];
						tauiceorig = ciwpmc[lay*ngptsw*ncol+ig*ncol+iplon] * extcoice[ig];
						taormc[lay*ngptsw*ncol+ig*ncol+iplon] = tauliqorig + tauiceorig;

						ssaliq = ssacoliq[ig] * (1.0 - forwliq[ig])/(1.0 - forwliq[ig] * ssacoliq[ig]);
						tauliq = (1.0 - forwliq[ig] * ssacoliq[ig]) * tauliqorig;
						ssaice = ssacoice[ig] * (1.0- forwice[ig]) /(1.0 - forwice[ig] * ssacoice[ig]);
						tauice = (1.0 - forwice[ig] * ssacoice[ig]) * tauiceorig;

						scatliq = ssaliq * tauliq;
						scatice = ssaice * tauice;
						taucmc[lay*ngptsw*ncol+ig*ncol+iplon] = tauliq + tauice;
					//Ensure non-zero taucmc and scatice
						if(taucmc[lay*ngptsw*ncol+ig*ncol+iplon]==0.0)
							taucmc[lay*ngptsw*ncol+ig*ncol+iplon] = cldmin;
						if(scatice==0.0)
							scatice = cldmin;

						ssacmc[lay*ngptsw*ncol+ig*ncol+iplon] = (scatliq + scatice) / taucmc[lay*ngptsw*ncol+ig*ncol+iplon];
			/*
			! In accordance with the 1996 Fu paper, equation A.3,
			! the moments for ice were calculated depending on whether using spheres
			! or hexagonal ice crystals.
			! Set asymetry parameter to first moment (istr=1)
			*/
						if (iceflag == 3)
						{
							istr = 1;
							asmcmc[lay*ngptsw*ncol+ig*ncol+iplon] = (1.0/(scatliq+scatice))*
								(scatliq*(pow(gliq[ig],istr) - forwliq[ig]) /
								(1.0 - forwliq[ig]) + scatice * pow((gice[ig]-forwice[ig])/
								(1.0 - forwice[ig]),istr));
						}
				// This code is the standard method for delta-m scaling.
				// Set asymetry parameter to first moment (istr=1)
						else
						{
							istr = 1;
							asmcmc[lay*ngptsw*ncol+ig*ncol+iplon] = (scatliq*
								(pow(gliq[ig],istr) - forwliq[ig]) /
								(1.0 - forwliq[ig]) + scatice * (pow(gice[ig],istr)-forwice[ig])/
								(1.0 - forwice[ig]))/(scatliq + scatice);

						}
					}
				}
	}
}


///////////////////////////////////////////////////////////////////////
//调用子例程 setcoef 来计算特定于该大气层的辐射传递例程所需的信息，尤其是通过对存储的参考大气中的数据进行插值来计算光学深度所需的一些系数和指标；
///////////////////////////////////////////////////////////////////////

__global__ void setcoef_sw1(double *tbound,double *tz,int *laytrop,
						int *layswtch,int *laylow,double *pavel,int *jp,
						double *preflog,int *jt,double *tavel,double *tref,
						int *jt1,double *wkl,double *coldry,double *forfac,
						int *indfor,double *forfrac,double *colh2o,double *colco2,
						double *colo3,double *coln2o,double *colch4,double *colo2,
						double *colmol,double *co2mult,double *selffac,double *selffrac,
						int *indself,double *fac10,double *fac00,double *fac11,double *fac01)
{
	//----- Input -----
	//----- Local -----
	int iplon,lay,jp1;

	double indbound,indlev0;

	double stpfac,tbndfrac,t0frac,plog;
	double fp,ft,ft1,water,scalefac,factor,co2reg,compfp;

	iplon=blockDim.x * blockIdx.x + threadIdx.x;
    lay=blockDim.y * blockIdx.y + threadIdx.y;
	/*
	Add one to nlayers here to include extra model layer at top of atmosphere
    Initialize all molecular amounts to zero here, then pass input amounts
    into RRTM array WKL below.
	*/
	stpfac=296.0/1013.0;
	if((iplon>=0&&iplon<ncol)&&(lay==0))
	{

		indbound=tbound[iplon]-159.0;
		tbndfrac=tbound[iplon]-int(tbound[iplon]);
		indlev0=tz[0*ncol+iplon]-159.0;
		t0frac=tz[0*ncol+iplon]-int(tz[0*ncol+iplon]);



		//a(iplon)=3
	}
		/*! Find the two reference pressures on either side of the
			layer pressure.  Store them in JP and JP1.  Store in FP the
			fraction of the difference (in ln(pressure)) between these
 			two values that the layer pressure lies.
		*/
		if((iplon>=0&&iplon<ncol)&&(lay>=0&&lay<nlayers))
		{
			plog=log(pavel[lay*ncol+iplon]);
			jp[lay*ncol+iplon]=int(36.0-5*(plog+0.04));
			if(jp[lay*ncol+iplon]<1)
				jp[lay*ncol+iplon]=1;
			else if(jp[lay*ncol+iplon]>58)
				jp[lay*ncol+iplon]=58;
			jp1=jp[lay*ncol+iplon]+1;
			fp=5.0*(preflog[jp[lay*ncol+iplon]-1]-plog);
			/*
			Determine, for each reference pressure (JP and JP1), which
			reference temperature (these are different for each
			reference pressure) is nearest the layer temperature but does
			not exceed it.  Store these indices in JT and JT1, resp.
			Store in FT (resp. FT1) the fraction of the way between JT
			(JT1) and the next highest reference temperature that the
			layer temperature falls.
			*/
			jt[lay*ncol+iplon]=int(3.0+(tavel[lay*ncol+iplon]-tref[jp[lay*ncol+iplon]-1])/15.0);
			if(jt[lay*ncol+iplon]<1)
				jt[lay*ncol+iplon]=1;
			else if(jt[lay*ncol+iplon]>4)
				jt[lay*ncol+iplon]=4;
			ft=((tavel[lay*ncol+iplon]-tref[jp[lay*ncol+iplon]-1])/15.0)-float(jt[lay*ncol+iplon]-3);
			jt1[lay*ncol+iplon]=int(3.0+(tavel[lay*ncol+iplon]-tref[jp1-1])/15.0);
			if(jt1[lay*ncol+iplon]<1)
				jt1[lay*ncol+iplon]=1;
			else if(jt1[lay*ncol+iplon]>4)
				jt1[lay*ncol+iplon]=4;
			ft1=((tavel[lay*ncol+iplon]-tref[jp1-1])/15.0)-float(jt1[lay*ncol+iplon]-3);

			water=wkl[lay*mxmol*ncol+0*ncol+iplon]/coldry[lay*ncol+iplon];
			scalefac = pavel[lay*ncol+iplon] * stpfac / tavel[lay*ncol+iplon];
		//If the pressure is less than ~100mb, perform a different
        //set of species interpolations.

			if(plog<=4.56)
			{
				//Set up factors needed to separately include the water vapor
				//foreign-continuum in the calculation of absorption coefficient.
				forfac[lay*ncol+iplon] = scalefac / (1.0+water);
				factor = (tavel[lay*ncol+iplon]-188.0)/36.0;
				indfor[lay*ncol+iplon] = 3;
				forfrac[lay*ncol+iplon] = factor - 1.0;

				//Calculate needed column amounts.
				colh2o[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+0*ncol+iplon];
				colco2[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+1*ncol+iplon];
				colo3[lay*ncol+iplon]  = (1.e-20) * wkl[lay*mxmol*ncol+2*ncol+iplon];
				coln2o[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+3*ncol+iplon];
				colch4[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+4*ncol+iplon];
				colo2[lay*ncol+iplon]  = (1.e-20) * wkl[lay*mxmol*ncol+6*ncol+iplon];
				colmol[lay*ncol+iplon] = (1.e-20) * coldry[lay*ncol+iplon] + colh2o[lay*ncol+iplon];
				if (colco2[lay*ncol+iplon] == 0.0)
					colco2[lay*ncol+iplon] = (1.e-32) * coldry[lay*ncol+iplon];
				if (coln2o[lay*ncol+iplon] == 0.0)
					coln2o[lay*ncol+iplon] = (1.e-32) * coldry[lay*ncol+iplon];
				if (colch4[lay*ncol+iplon] == 0.0)
					colch4[lay*ncol+iplon] = (1.e-32) * coldry[lay*ncol+iplon];
				if (colo2[lay*ncol+iplon]  == 0.0)
					colo2[lay*ncol+iplon]  = (1.e-32) * coldry[lay*ncol+iplon];
				co2reg = (3.55e-24) * coldry[lay*ncol+iplon];
				co2mult[lay*ncol+iplon]= (colco2[lay*ncol+iplon] - co2reg) *
					   272.63 * exp(-1919.4/tavel[lay*ncol+iplon])/((8.7604e-4)*tavel[lay*ncol+iplon]);

					selffac[lay*ncol+iplon]= 0.0;
					selffrac[lay*ncol+iplon]= 0.0;
					indself[lay*ncol+iplon] = 0;
			}
			else
			{
				//Set up factors needed to separately include the water vapor
				//foreign-continuum in the calculation of absorption coefficient.

				forfac[lay*ncol+iplon] = scalefac / (1.0+water);
				factor = (332.0-tavel[lay*ncol+iplon])/36.0;
				indfor[lay*ncol+iplon] = min(2, max(1, int(factor)));
				forfrac[lay*ncol+iplon] = factor - float(indfor[lay*ncol+iplon]);

				//Set up factors needed to separately include the water vapor
				//self-continuum in the calculation of absorption coefficient.

				selffac[lay*ncol+iplon] = water * forfac[lay*ncol+iplon];
				factor = (tavel[lay*ncol+iplon]-188.0)/7.2;
				indself[lay*ncol+iplon] = min(9, max(1, int(factor)-7));
				selffrac[lay*ncol+iplon] = factor - float(indself[lay*ncol+iplon] + 7);

				//Calculate needed column amounts.

				colh2o[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+0*ncol+iplon];
				colco2[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+1*ncol+iplon];
				colo3[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+2*ncol+iplon];
				coln2o[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+3*ncol+iplon];
				colch4[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+5*ncol+iplon];
				colo2[lay*ncol+iplon] = (1.e-20) * wkl[lay*mxmol*ncol+6*ncol+iplon];
				colmol[lay*ncol+iplon] = (1.e-20) * coldry[lay*ncol+iplon] + colh2o[lay*ncol+iplon];

				if (colco2[lay*ncol+iplon] == 0.0)
					colco2[lay*ncol+iplon] = (1.e-32) * coldry[lay*ncol+iplon];
				if (coln2o[lay*ncol+iplon]== 0.0)
					coln2o[lay*ncol+iplon] = (1.e-32) * coldry[lay*ncol+iplon];
				if (colch4[lay*ncol+iplon] == 0.0)
					colch4[lay*ncol+iplon] = (1.e-32) * coldry[lay*ncol+iplon];
				if (colo2[lay*ncol+iplon] == 0.0)
					colo2[lay*ncol+iplon] = (1.e-32) * coldry[lay*ncol+iplon];
				//Using E = 1334.2 cm-1.
				co2reg = (3.55e-24) * coldry[lay*ncol+iplon];
				co2mult[lay*ncol+iplon]= (colco2[lay*ncol+iplon] - co2reg) *
						272.63 * exp(-1919.4/tavel[lay*ncol+iplon])/((8.7604e-4)*tavel[lay*ncol+iplon]);

			}
			//5400
			/*
			 We have now isolated the layer ln pressure and temperature,
			between two reference pressures and two reference temperatures
			(for each reference pressure).  We multiply the pressure
			fraction FP with the appropriate temperature fractions to get
			the factors that will be needed for the interpolation that yields
			the optical depths (performed in routines TAUGBn for band n).
			*/

			compfp = 1.0 - fp;
			fac10[lay*ncol+iplon] = compfp * ft;
			fac00[lay*ncol+iplon] = compfp * (1.0 - ft);
			fac11[lay*ncol+iplon] = fp * ft1;
			fac01[lay*ncol+iplon] = fp * (1.0 - ft1);

		}
		return;

}
__global__ void setcoef_sw2(int *laytrop,
						    int *layswtch,int *laylow,double *pavel)
{
    int iplon,lay;
    double plog2;
    iplon=blockDim.x * blockIdx.x + threadIdx.x;
    laytrop[iplon]=0;
    if(iplon>=0&&iplon<ncol)
    {
        laytrop[iplon] = 0;
        layswtch[iplon] = 0;
        laylow[iplon] = 0;
        for(lay=0;lay<nlayers;lay++)
        {
            plog2=log(pavel[lay*ncol+iplon]);
            if(plog2>4.56)
            {
                laytrop[iplon] ++;
                if(plog2>=6.62)
                laylow[iplon] ++;
            }
        }
    }
    return;
}


///////////////////////////////////////////////////////////////////////
//调用子程序taumol 来计算16个光谱带中每一个的气态光学深度和普朗克分数
///////////////////////////////////////////////////////////////////////

__global__ void taumol_sw(double oneminus,int *laytrop,double *colh2o,double *colch4,
						double *fac00,double *fac10,double *fac01,double *fac11,
						int *jp,int *jt,int *nspa,int *jt1,int *indself,int *indfor,
						double *colmol,double *ztaug,double *absa16,double *selffac,
						double *selfref16,double *selffrac,double *forfac,double *forref16,
						double *forfrac,double *ztaur,int *nspb,double *absb16,double *zsflxzen,
						double *sfluxref16,double *colco2,double *absa17,double *selfref17,
						double *forref17,double *absb17,double *sfluxref17,double *absa18,
						double *selfref18,double *forref18,double *sfluxref18,double *absb18,
						double *absa19,double *selfref19,double *forref19,double *sfluxref19,
						double *absb19,double *absa20,double *selfref20,double *forref20,
						double *absch420,double *sfluxref20,double *absb20,double *absa21,
						double *selfref21,double *forref21,double *sfluxref21,double *absb21,
						double *colo2,double *absa22,double *selfref22,double *forref22,
						double *sfluxref22,double *absb22,double *rayl23,double *absa23,
						double *selfref23,double *forref23,double *sfluxref23,double *rayla24,
						double *absa24,double *colo3,double *abso3a24,double *selfref24,
						double *forref24,double *sfluxref24,double *absb24,double *abso3b24,
						double *rayl25,double *absa25,double *abso3a25,double *sfluxref25,
						double *sfluxref26,double *rayl26,double *rayl27,double *absa27,
						double *absb27,double *sfluxref27,double *absa28,double *absb28,
						double *sfluxref28,double *absa29,double *selfref29,double *forref29,
						double *absco229,double *absb29,double *absh2o29,double *sfluxref29,
						double *abso3b25,double *raylb24)
{
	int iplon,lay,ig;
	iplon=blockDim.x * blockIdx.x + threadIdx.x;
    lay=blockDim.y * blockIdx.y + threadIdx.y;
	//--------------taumol16---------------------
	int ind0_16, ind1_16, inds_16, indf_16, js_16, laysolfr_16;
    double fac000_16, fac001_16, fac010_16, fac011_16, fac100_16, fac101_16,fac110_16, fac111_16;
	double fs_16, speccomb_16, specmult_16, specparm_16, tauray_16;

	const int ng16 = 6;
    const int layreffr_16=30;
    double rayl_16, strrat1_16;

	rayl_16=1.0;
	strrat1_16=1.0;

	//--------------taumol17---------------------
	int  ind0_17, ind1_17, inds_17, indf_17, js_17,laysolfr_17;
    double fac000_17, fac001_17, fac010_17, fac011_17, fac100_17, fac101_17,fac110_17, fac111_17;
	double fs_17, speccomb_17, specmult_17, specparm_17, tauray_17;

	const int ng17 = 12;
    const int ngs16 = 6;

	const int layreffr_17=30;
    double rayl_17, strrat_17;

	rayl_17=1.0;
	strrat_17=1.0;

	//-------------taumol18----------------------
	int ind0_18, ind1_18, inds_18, indf_18, js_18, laysolfr_18;
    double fac000_18, fac001_18, fac010_18, fac011_18, fac100_18, fac101_18,fac110_18, fac111_18;
	double fs_18, speccomb_18, specmult_18, specparm_18,tauray_18;

    const int  ng18 = 8;
    const int ngs17 = 18;

	const int layreffr_18=30;
    double rayl_18, strrat_18;

	rayl_18=1.0;
    strrat_18=1.0;

	//-------------taumol19----------------------
	int ind0_19, ind1_19, inds_19, indf_19, js_19, laysolfr_19;
    double fac000_19, fac001_19, fac010_19, fac011_19, fac100_19, fac101_19, fac110_19, fac111_19;
	double fs_19, speccomb_19, specmult_19, specparm_19,tauray_19;

    const int ng19 = 8;
    const int ngs18 = 26;
    const int layreffr_19=30;
    double rayl_19, strrat_19;

	rayl_19=1.0;
	strrat_19=1.0;

	//-------------taumol20----------------------
	int  ind0_20, ind1_20, inds_20, indf_20,  laysolfr_20;
    double tauray_20;

    const int ng20 = 10;
    const int ngs19 = 34;
	const int layreffr_20=30;
    double rayl_20;

	rayl_20=1.0;

	//-------------taumol21----------------------
	int ind0_21, ind1_21, inds_21, indf_21, js_21,laysolfr_21;
    double fac000_21, fac001_21, fac010_21, fac011_21, fac100_21, fac101_21,fac110_21, fac111_21;
	double fs_21, speccomb_21, specmult_21, specparm_21, tauray_21;

	const int ng21 = 10;
    const int ngs20 = 44;
    const int layreffr_21=30;
    double rayl_21, strrat_21;

	rayl_21=1.0;
	strrat_21=1.0;

	//-------------taumol22----------------------
	int  ind0_22, ind1_22, inds_22, indf_22, js_22, laysolfr_22;
    double fac000_22, fac001_22, fac010_22, fac011_22, fac100_22, fac101_22,fac110_22, fac111_22;
	double	fs_22, speccomb_22, specmult_22, specparm_22,tauray_22, o2adj_22, o2cont_22;

    const int ng22 = 2;
    const int ngs21 = 54;
    const int layreffr_22=30;
    double rayl_22, strrat_22;

	rayl_22=1.0;
	strrat_22=1.0;

	o2adj_22=1.0;

	//-------------taumol23----------------------

	int ind0_23, ind1_23, inds_23, indf_23,  laysolfr_23;
    double tauray_23;

    const int ngs22 = 56;
    const int ng23 = 10;
    const int layreffr_23=30;
    double givfac_23;

	givfac_23=1.0;

	//-------------taumol24----------------------

    int ind0_24, ind1_24, inds_24, indf_24, js_24, laysolfr_24;
    double fac000_24, fac001_24, fac010_24, fac011_24, fac100_24, fac101_24,fac110_24, fac111_24;
	double fs_24, speccomb_24, specmult_24, specparm_24,tauray_24;

	const int ng24 = 8;
    const int ngs23 = 66;
    const int layreffr_24=30;
    double strrat_24;

	 strrat_24=1.0;

	//-------------taumol25----------------------
	int ind0_25,ind1_25,laysolfr_25;
    double  tauray_25;

    const int ng25 = 6;
    const int ngs24 = 74;
    const int layreffr_25=30;

	//-------------taumol26----------------------
	int ind0_26, ind1_26, laysolfr_26;

    const int ng26 = 6;
	const int ngs25 = 80;

	//-------------taumol27----------------------
	int ind0_27, ind1_27,laysolfr_27;
    double  tauray_27;

    const int ng27 = 8;
    const int ngs26 = 86;
    const int layreffr_27=30;
    double scalekur_27;

	scalekur_27=1.0;

	//-------------taumol28----------------------
	int  ind0_28, ind1_28, js_28,laysolfr_28;
    double fac000_28, fac001_28, fac010_28,fac011_28, fac100_28, fac101_28,fac110_28, fac111_28;
	double fs_28, speccomb_28, specmult_28, specparm_28,tauray_28;

	const int ngs27 = 94;
    const int ng28 = 6;
	const int layreffr_28=30;
    double rayl_28, strrat_28;

	rayl_28=1.0;
	strrat_28=1.0;
	//-------------taumol29----------------------
	int  ind0_29, ind1_29, inds_29, indf_29,laysolfr_29;
    double tauray_29;

	const int ng29 = 12;
    const int ngs28 = 100;
    const int layreffr_29=30;
    double rayl_29;

	//******************taumol16****************
		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			speccomb_16 = colh2o[lay*ncol+iplon] + strrat1_16*colch4[lay*ncol+iplon];
			specparm_16 = colh2o[lay*ncol+iplon]/speccomb_16 ;

			if (specparm_16 >= oneminus) specparm_16 = oneminus;
			specmult_16 = 8.0*(specparm_16);
			js_16 = 1 + (int)(specmult_16);
			fs_16 = specmult_16 - (int)specmult_16;
			fac000_16 = (1.0 - fs_16) * fac00[lay*ncol+iplon];
			fac010_16 = (1.0 - fs_16) * fac10[lay*ncol+iplon];
			fac100_16 = fs_16 * fac00[lay*ncol+iplon];
			fac110_16 = fs_16 * fac10[lay*ncol+iplon];
			fac001_16 = (1.0 - fs_16) * fac01[lay*ncol+iplon];
			fac011_16 = (1.0 - fs_16) * fac11[lay*ncol+iplon];
			fac101_16 = fs_16 * fac01[lay*ncol+iplon];
			fac111_16 = fs_16 * fac11[lay*ncol+iplon];
			ind0_16 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[0] + js_16;
			ind1_16 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[0] + js_16;
			inds_16 = indself[lay*ncol+iplon];
			indf_16 = indfor[lay*ncol+iplon];
			tauray_16 = colmol[lay*ncol+iplon] * rayl_16;

			if(iplon==0)
					printf("ind0_16=%d",ind0_16);
			for(ig=0;ig<ng16;ig++)
			{
				ztaug[ig*nlayers*ncol+lay*ncol+iplon] = speccomb_16 *
					(fac000_16 * absa16[ig*585+ind0_16 -1]  +
					fac100_16 * absa16[ig*585+ind0_16]  +
					fac010_16 * absa16[ig*585+ind0_16 +8] +
					fac110_16 * absa16[ig*585+ind0_16 +9] +
					fac001_16 * absa16[ig*585+ind1_16 -1] +
					fac101_16 * absa16[ig*585+ind1_16] +
					fac011_16 * absa16[ig*585+ind1_16 +8] +
					fac111_16 * absa16[ig*585+ind1_16 +9]) +
					colh2o[lay*ncol+iplon] *
					(selffac[lay*ncol+iplon] * (selfref16[ig*10+inds_16 - 1] +
					selffrac[lay*ncol+iplon] *
					(selfref16[ig*10+inds_16] - selfref16[ig*10+inds_16 - 1])) +
					forfac[lay*ncol+iplon] * (forref16[ig*3+indf_16 - 1] +
					forfrac[lay*ncol+iplon] *
					(forref16[ig*3+indf_16] - forref16[ig*3+indf_16 - 1])));
			// ssa(lay,ig) = tauray/ztaug(lay,ig)
				ztaur[ig*nlayers*ncol+lay*ncol+iplon] = tauray_16;

			}
		}

		laysolfr_16 = nlayers;

// Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			if (jp[(lay-1)*ncol+iplon] < layreffr_16 && jp[lay*ncol+iplon] >= layreffr_16)
				laysolfr_16 = lay+1;
			ind0_16 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[0] + 1;
			ind1_16 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[0]+ 1;

			/*****************************************************/
			tauray_16 = colmol[lay*ncol+iplon] * rayl_16;

			for(ig=0;ig<ng16;ig++)
			{
				ztaug[ig*nlayers*ncol+lay*ncol+iplon] = colch4[lay*ncol+iplon] *
					(fac00[lay*ncol+iplon] * absb16[ig*235+ind0_16 - 1] +
					fac10[lay*ncol+iplon]* absb16[ig*235+ind0_16] +
					fac01[lay*ncol+iplon] * absb16[ig*235+ind1_16 - 1] +
					fac11[lay*ncol+iplon] * absb16[ig*235+ind1_16]);

				if (lay == laysolfr_16-1) zsflxzen[ig*ncol+iplon] = sfluxref16[ig];
				ztaur[ig*nlayers*ncol+lay*ncol+iplon] = tauray_16 ;
			}
		}

	//******************taumol17****************
        if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			speccomb_17 = colh2o[lay*ncol+iplon] + strrat_17*colco2[lay*ncol+iplon];
			specparm_17 = colh2o[lay*ncol+iplon]/speccomb_17;

			if (specparm_17 >= oneminus) specparm_17 = oneminus;
			specmult_17 = 8.0*(specparm_17);
			js_17 = 1 + (int)(specmult_17);
			fs_17 = specmult_17 - (int)specmult_17;
			fac000_17 = (1.0 - fs_17) * fac00[lay*ncol+iplon];
			fac010_17 = (1.0 - fs_17) * fac10[lay*ncol+iplon];
			fac100_17 = fs_17 * fac00[lay*ncol+iplon];
			fac110_17 = fs_17 * fac10[lay*ncol+iplon];
			fac001_17 = (1.0 - fs_17) * fac01[lay*ncol+iplon];
			fac011_17 = (1.0 - fs_17) * fac11[lay*ncol+iplon];
			fac101_17 = fs_17 * fac01[lay*ncol+iplon];
			fac111_17 = fs_17 * fac11[lay*ncol+iplon];
			ind0_17 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[1] + js_17;
			ind1_17 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[1] + js_17;
			inds_17 = indself[lay*ncol+iplon];
			indf_17 = indfor[lay*ncol+iplon];
			tauray_17 = colmol[lay*ncol+iplon] * rayl_17;

			for(ig=0;ig<ng17;ig++)
			{
				ztaug[(ngs16+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_17 *
					(fac000_17 * absa17[ig*585+ind0_17-1] +
					fac100_17 * absa17[ig*585+ind0_17] +
					fac010_17 * absa17[ig*585+ind0_17+8] +
					fac110_17 * absa17[ig*585+ind0_17+9] +
					fac001_17 * absa17[ig*585+ind1_17-1] +
					fac101_17 * absa17[ig*585+ind1_17] +
					fac011_17 * absa17[ig*585+ind1_17+8] +
					fac111_17 * absa17[ig*585+ind1_17+9]) +
					colh2o[lay*ncol+iplon] *
					(selffac[lay*ncol+iplon] * (selfref17[ig*10+inds_17-1] +
					selffrac[lay*ncol+iplon] *
					(selfref17[ig*10+inds_17] - selfref17[ig*10+inds_17-1])) +
					forfac[lay*ncol+iplon] * (forref17[ig*4+indf_17-1] +
					forfrac[lay*ncol+iplon] *
					(forref17[ig*4+indf_17] - forref17[ig*4+indf_17-1])));

				ztaur[(ngs16+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_17;
			}
		}

		laysolfr_17 = nlayers;

// Upper atmosphere loop
        if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			if ((jp[(lay-1)*ncol+iplon] < layreffr_17 )&& (jp[lay*ncol+iplon] >= layreffr_17))
				laysolfr_17 = lay+1;
			speccomb_17 = colh2o[lay*ncol+iplon] + strrat_17*colco2[lay*ncol+iplon];
			specparm_17 = colh2o[lay*ncol+iplon]/speccomb_17 ;

			if (specparm_17 >= oneminus) specparm_17 = oneminus;
			specmult_17 = 4.0*(specparm_17);
			js_17 = 1 + (int)(specmult_17);
			fs_17 = specmult_17 - (int)specmult_17;
			fac000_17 = (1.0 - fs_17) * fac00[lay*ncol+iplon];
			fac010_17 = (1.0 - fs_17) * fac10[lay*ncol+iplon];
			fac100_17 = fs_17 * fac00[lay*ncol+iplon];
			fac110_17 = fs_17 * fac10[lay*ncol+iplon];
			fac001_17 = (1.0 - fs_17) * fac01[lay*ncol+iplon];
			fac011_17 = (1.0 - fs_17) * fac11[lay*ncol+iplon];
			fac101_17 = fs_17 * fac01[lay*ncol+iplon];
			fac111_17 = fs_17 * fac11[lay*ncol+iplon];
			ind0_17 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[1] + js_17;
			ind1_17 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[1] + js_17;
			indf_17 = indfor[lay*ncol+iplon];
			tauray_17 = colmol[lay*ncol+iplon] * rayl_17;

			for(ig=0;ig<ng17;ig++)
			{
				ztaug[(ngs16+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_17 *
					(fac000_17 * absb17[ig*1175+ind0_17-1] +
					fac100_17 * absb17[ig*1175+ind0_17] +
					fac010_17 * absb17[ig*1175+ind0_17+4] +
					fac110_17 * absb17[ig*1175+ind0_17+5] +
					fac001_17 * absb17[ig*1175+ind1_17-1] +
					fac101_17 * absb17[ig*1175+ind1_17] +
					fac011_17 * absb17[ig*1175+ind1_17+4] +
					fac111_17 * absb17[ig*1175+ind1_17+5]) +
					colh2o[lay*ncol+iplon] *
					forfac[lay*ncol+iplon] * (forref17[ig*4+indf_17-1] +
					forfrac[lay*ncol+iplon] *
					(forref17[ig*4+indf_17] - forref17[ig*4+indf_17-1]));

				if (lay == laysolfr_17 - 1) zsflxzen[(ngs16+ig)*ncol+iplon] = sfluxref17[(js_17-1)*12+ig]
					+ fs_17 * (sfluxref17[js_17*12+ig] - sfluxref17[(js_17-1)*12+ig]);
				ztaur[(ngs16+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_17;
			}
		}


	//******************taumol18****************

		laysolfr_18 = laytrop[iplon];
		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			if ((jp[lay*ncol+iplon] < layreffr_18) && (jp[(lay+1)*ncol+iplon] >= layreffr_18))
				laysolfr_18 = (((lay+1)<laytrop[iplon])?(lay+1):laytrop[iplon]);
			speccomb_18 = colh2o[lay*ncol+iplon] + strrat_18*colch4[lay*ncol+iplon];
			specparm_18 = colh2o[lay*ncol+iplon]/speccomb_18 ;

			if (specparm_18 >= oneminus) specparm_18 = oneminus;
			specmult_18 = 8.0*(specparm_18);
			js_18 = 1 + (int)(specmult_18);
			fs_18 = specmult_18 - (int)specmult_18;
			fac000_18 = (1.0 - fs_18) * fac00[lay*ncol+iplon];
			fac010_18 = (1.0 - fs_18) * fac10[lay*ncol+iplon];
			fac100_18 = fs_18 * fac00[lay*ncol+iplon];
			fac110_18 = fs_18 * fac10[lay*ncol+iplon];
			fac001_18 = (1.0 - fs_18) * fac01[lay*ncol+iplon];
			fac011_18 = (1.0 - fs_18) * fac11[lay*ncol+iplon];
			fac101_18 = fs_18 * fac01[lay*ncol+iplon];
			fac111_18 = fs_18 * fac11[lay*ncol+iplon];
			ind0_18 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[2]+ js_18;
			ind1_18 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[2] + js_18;
			inds_18 = indself[lay*ncol+iplon];
			indf_18 = indfor[lay*ncol+iplon];
			tauray_18 = colmol[lay*ncol+iplon] * rayl_18;

			for(ig=0;ig<ng18;ig++)
			{
				ztaug[(ngs17+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_18 *
					(fac000_18 * absa18[ig*585+ind0_18-1] +
					fac100_18 * absa18[ig*585+ind0_18] +
					fac010_18 * absa18[ig*585+ind0_18+8] +
					fac110_18 * absa18[ig*585+ind0_18+9] +
					fac001_18 * absa18[ig*585+ind1_18-1] +
					fac101_18 * absa18[ig*585+ind1_18] +
					fac011_18 * absa18[ig*585+ind1_18+8] +
					fac111_18 * absa18[ig*585+ind1_18+9]) +
					colh2o[lay*ncol+iplon] *
					(selffac[lay*ncol+iplon] * (selfref18[ig*10+inds_18-1] +
					selffrac[lay*ncol+iplon] *
					(selfref18[ig*10+inds_18] - selfref18[ig*10+inds_18-1])) +
					forfac[lay*ncol+iplon] * (forref18[ig*3+indf_18-1] +
					forfrac[lay*ncol+iplon] *
					(forref18[ig*3+indf_18] - forref18[ig*3+indf_18-1])));

				if (lay == laysolfr_18 - 1) zsflxzen[(ngs17+ig)*ncol+iplon] = sfluxref18[(js_18-1)*8+ig]
				+ fs_18 * (sfluxref18[(js_18)*8+ig] - sfluxref18[(js_18-1)*8+ig]);
				ztaur[(ngs17+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_18;
			}
		}

// Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			ind0_18 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[2] + 1;
			ind1_18 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[2] + 1;
			tauray_18 = colmol[lay*ncol+iplon]* rayl_18;

			for(ig = 0;ig< ng18;ig++)
			{
				ztaug[(ngs17+ig)*nlayers*ncol+lay*ncol+iplon] = colch4[lay*ncol+iplon] *
					(fac00[lay*ncol+iplon] * absb18[ig*235+ind0_18-1] +
					fac10[lay*ncol+iplon] * absb18[ig*235+ind0_18] +
					fac01[lay*ncol+iplon] * absb18[ig*235+ind1_18-1] +
					fac11[lay*ncol+iplon] * absb18[ig*235+ind1_18]) ;
			ztaur[(ngs17+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_18;
			}
		}

    //******************taumol19****************

		laysolfr_19 = laytrop[iplon];

		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			if ((jp[lay*ncol+iplon] < layreffr_19) && (jp[(lay+1)*ncol+iplon] >= layreffr_19))
				laysolfr_19 = (((lay+1)<laytrop[iplon])?(lay+1):laytrop[iplon]);
			speccomb_19 = colh2o[lay*ncol+iplon] + strrat_19*colco2[lay*ncol+iplon];
			specparm_19 = colh2o[lay*ncol+iplon]/speccomb_19;

			if (specparm_19 >= oneminus) specparm_19 = oneminus;
			specmult_19 = 8.0*(specparm_19);
			js_19 = 1 + (int)(specmult_19);
			fs_19 =specmult_19 - (int)specmult_19;
			fac000_19 = (1.0 - fs_19) * fac00[lay*ncol+iplon];
			fac010_19 = (1.0 - fs_19) * fac10[lay*ncol+iplon];
			fac100_19 = fs_19 * fac00[lay*ncol+iplon];
			fac110_19 = fs_19 * fac10[lay*ncol+iplon];
			fac001_19 = (1.0 - fs_19) * fac01[lay*ncol+iplon];
			fac011_19 = (1.0 - fs_19) * fac11[lay*ncol+iplon];
			fac101_19 = fs_19 * fac01[lay*ncol+iplon];
			fac111_19 = fs_19 * fac11[lay*ncol+iplon];
			ind0_19 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[3] + js_19;
			ind1_19 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[3] + js_19;
			inds_19 = indself[lay*ncol+iplon];
			indf_19 = indfor[lay*ncol+iplon];
			tauray_19 = colmol[lay*ncol+iplon] * rayl_19;

			for(ig = 0;ig< ng19;ig++)
			{
				ztaug[(ngs18+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_19 *
					(fac000_19 * absa19[ig*585+ind0_19-1] +
					fac100_19 * absa19[ig*585+ind0_19] +
					fac010_19 * absa19[ig*585+ind0_19+8] +
					fac110_19 * absa19[ig*585+ind0_19+9] +
					fac001_19 * absa19[ig*585+ind1_19-1] +
					fac101_19 * absa19[ig*585+ind1_19] +
					fac011_19 * absa19[ig*585+ind1_19+8] +
					fac111_19 * absa19[ig*585+ind1_19+9]) +
					colh2o[lay*ncol+iplon] *
					(selffac[lay*ncol+iplon] * (selfref19[ig*10+inds_19-1] +
					selffrac[lay*ncol+iplon] *
					(selfref19[ig*10+inds_19] - selfref19[ig*10+inds_19-1])) +
					forfac[lay*ncol+iplon] * (forref19[ig*3+indf_19-1] +
					forfrac[lay*ncol+iplon] *
					(forref19[ig*3+indf_19] - forref19[ig*3+indf_19-1]))) ;
				if (lay == laysolfr_19 - 1) zsflxzen[(ngs18+ig)*ncol+iplon] = sfluxref19[(js_19-1)*8+ig]
					+ fs_19 * (sfluxref19[js_19*8+ig] - sfluxref19[(js_19-1)*8+ig]);
				ztaur[(ngs18+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_19;
			}
		}

// Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			ind0_19 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[3] + 1;
			ind1_19 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[3] + 1;
			tauray_19 = colmol[lay*ncol+iplon] * rayl_19;

			for(ig =0;ig<ng19;ig++)
			{
				ztaug[(ngs18+ig)*nlayers*ncol+lay*ncol+iplon] = colco2[lay*ncol+iplon] *
					(fac00[lay*ncol+iplon] * absb19[ig*235+ind0_19-1] +
					fac10[lay*ncol+iplon] * absb19[ig*235+ind0_19] +
					fac01[lay*ncol+iplon] * absb19[ig*235+ind1_19-1] +
					fac11[lay*ncol+iplon] * absb19[ig*235+ind1_19]) ;

				ztaur[(ngs18+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_19 ;
			}
		}

	//******************taumol20****************

		laysolfr_20 = laytrop[iplon];
// Lower atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			if ((jp[lay*ncol+iplon] < layreffr_20) && (jp[(lay+1)*ncol+iplon] >= layreffr_20))
				laysolfr_20 =(((lay+1)<laytrop[iplon])?(lay+1):laytrop[iplon]);
			ind0_20 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[4] + 1;
			ind1_20 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[4] + 1;
			inds_20 = indself[lay*ncol+iplon];
			indf_20 = indfor[lay*ncol+iplon];
			tauray_20 = colmol[lay*ncol+iplon] * rayl_20;

			for(ig = 0;ig< ng20;ig++)
			{
				ztaug[(ngs19+ig)*nlayers*ncol+lay*ncol+iplon] = colh2o[lay*ncol+iplon] *
					((fac00[lay*ncol+iplon] * absa20[ig*65+ind0_20-1] +
					fac10[lay*ncol+iplon] * absa20[ig*65+ind0_20] +
					fac01[lay*ncol+iplon] * absa20[ig*65+ind1_20-1] +
					fac11[lay*ncol+iplon] * absa20[ig*65+ind1_20]) +
					selffac[lay*ncol+iplon] * (selfref20[ig*10+inds_20-1] +
					selffrac[lay*ncol+iplon] *
					(selfref20[ig*10+inds_20] - selfref20[ig*10+inds_20-1])) +
					forfac[lay*ncol+iplon] * (forref20[ig*4+indf_20-1] +
					forfrac[lay*ncol+iplon] *
					(forref20[ig*4+indf_20] - forref20[ig*4+indf_20-1])))
					+ colch4[lay*ncol+iplon] * absch420[ig];
				ztaur[(ngs19+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_20;
				if (lay == laysolfr_20 - 1) zsflxzen[(ngs19+ig)*ncol+iplon] = sfluxref20[ig];
			}
		}

// Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			ind0_20 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[4] + 1;
			ind1_20 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[4] + 1;
			indf_20 = indfor[lay*ncol+iplon];
			tauray_20 = colmol[lay*ncol+iplon] * rayl_20;

			for(ig=0;ig< ng20;ig++)
			{
				ztaug[(ngs19+ig)*nlayers*ncol+lay*ncol+iplon]= colh2o[lay*ncol+iplon] *
					(fac00[lay*ncol+iplon] * absb20[ig*235+ind0_20-1] +
					fac10[lay*ncol+iplon]* absb20[ig*235+ind0_20] +
					fac01[lay*ncol+iplon] * absb20[ig*235+ind1_20-1] +
					fac11[lay*ncol+iplon] * absb20[ig*235+ind1_20] +
					forfac[lay*ncol+iplon] * (forref20[ig*4+indf_20-1] +
					forfrac[lay*ncol+iplon] *
					(forref20[ig*4+indf_20] - forref20[ig*4+indf_20-1]))) +
					colch4[lay*ncol+iplon] * absch420[ig];
				//ssa(lay,ngs19+ig) = tauray_20/ztaug(lay,ngs19+ig)
				ztaur[(ngs19+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_20;
			}
		}

	//******************taumol21****************

		laysolfr_21 = laytrop[iplon];
// Lower atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			if ((jp[lay*ncol+iplon] < layreffr_21) && (jp[(lay+1)*ncol+iplon] >= layreffr_21))
				laysolfr_21 = (((lay+1)<laytrop[iplon])?(lay+1):laytrop[iplon]);
			speccomb_21 = colh2o[lay*ncol+iplon] + strrat_21*colco2[lay*ncol+iplon];
			specparm_21 = colh2o[lay*ncol+iplon]/speccomb_21;

			  if (specparm_21 >= oneminus) specparm_21 = oneminus;
			 specmult_21 = 8.0*(specparm_21);
			 js_21 = 1 + (int)(specmult_21);
			 fs_21 =specmult_21 - (int)specmult_21;
			 fac000_21 = (1.0 - fs_21) * fac00[lay*ncol+iplon];
			 fac010_21 = (1.0 - fs_21) * fac10[lay*ncol+iplon];
			 fac100_21 = fs_21 * fac00[lay*ncol+iplon];
			 fac110_21 = fs_21 * fac10[lay*ncol+iplon];
			 fac001_21 = (1.0 - fs_21) * fac01[lay*ncol+iplon];
			 fac011_21 = (1.0 - fs_21) * fac11[lay*ncol+iplon];
			 fac101_21 = fs_21 * fac01[lay*ncol+iplon];
			 fac111_21 = fs_21 * fac11[lay*ncol+iplon];
			 ind0_21 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[5] + js_21;
			 ind1_21 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[5] + js_21;
			 inds_21 = indself[lay*ncol+iplon];
			 indf_21 = indfor[lay*ncol+iplon];
			 tauray_21 = colmol[lay*ncol+iplon] * rayl_21;

			 for(ig =0;ig< ng21;ig++)
			 {
				ztaug[(ngs20+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_21 *
					(fac000_21 * absa21[ig*585+ind0_21-1] +
					 fac100_21 * absa21[ig*585+ind0_21] +
					 fac010_21 * absa21[ig*585+ind0_21+8] +
					 fac110_21 * absa21[ig*585+ind0_21+9] +
					 fac001_21 * absa21[ig*585+ind1_21-1]+
					 fac101_21 * absa21[ig*585+ind1_21] +
					 fac011_21 * absa21[ig*585+ind1_21+8] +
					 fac111_21 * absa21[ig*585+ind1_21+9]) +
					 colh2o[lay*ncol+iplon] *
					 (selffac[lay*ncol+iplon] * (selfref21[ig*10+inds_21-1] +
					 selffrac[lay*ncol+iplon] *
					 (selfref21[ig*10+inds_21] - selfref21[ig*10+inds_21-1])) +
					 forfac[lay*ncol+iplon] * (forref21[ig*4+indf_21-1] +
					 forfrac[lay*ncol+iplon] *
					 (forref21[ig*4+indf_21] - forref21[ig*4+indf_21-1])));
				//ssa(lay,ngs20+ig) = tauray_21/ztaug(lay,ngs20+ig)
				if (lay == laysolfr_21 - 1) zsflxzen[(ngs20+ig)*ncol+iplon] = sfluxref21[(js_21-1)*10+ig]
				   + fs_21 * (sfluxref21[js_21*10+ig] - sfluxref21[(js_21-1)*10+ig]);
				ztaur[(ngs20+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_21;
			}
		 }

// Upper atmosphere loop
		 if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		 {
			 speccomb_21 = colh2o[lay*ncol+iplon] + strrat_21*colco2[lay*ncol+iplon];
			 specparm_21 = colh2o[lay*ncol+iplon]/speccomb_21;
			 if (specparm_21 >= oneminus) specparm_21 = oneminus;
			 specmult_21 = 4.0*(specparm_21);
			 js_21 = 1 + (int)(specmult_21);
			 fs_21 = specmult_21 - (int)specmult_21;
			 fac000_21 = (1.0 - fs_21) * fac00[lay*ncol+iplon];
			 fac010_21 = (1.0 - fs_21) * fac10[lay*ncol+iplon];
			 fac100_21 = fs_21 * fac00[lay*ncol+iplon];
			 fac110_21 = fs_21 * fac10[lay*ncol+iplon];
			 fac001_21 = (1.0 - fs_21) * fac01[lay*ncol+iplon];
			 fac011_21 = (1.0 - fs_21) * fac11[lay*ncol+iplon];
			 fac101_21 = fs_21 * fac01[lay*ncol+iplon];
			 fac111_21 = fs_21 * fac11[lay*ncol+iplon];
			 ind0_21 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[5] + js_21;
			 ind1_21 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[5] + js_21;
			 indf_21 = indfor[lay*ncol+iplon];
			 tauray_21 = colmol[lay*ncol+iplon] * rayl_21;

			 for(ig = 0;ig< ng21;ig++)
			 {
				ztaug[(ngs20+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_21 *
					(fac000_21 * absb21[ig*1175+ind0_21-1] +
					 fac100_21 * absb21[ig*1175+ind0_21] +
					 fac010_21 * absb21[ig*1175+ind0_21+4] +
					 fac110_21 * absb21[ig*1175+ind0_21+5] +
					 fac001_21 * absb21[ig*1175+ind1_21-1] +
					 fac101_21 * absb21[ig*1175+ind1_21] +
					 fac011_21 * absb21[ig*1175+ind1_21+4] +
					 fac111_21* absb21[ig*1175+ind1_21+5]) +
					 colh2o[lay*ncol+iplon] *
					 forfac[lay*ncol+iplon] * (forref21[ig*4+indf_21-1] +
					 forfrac[lay*ncol+iplon] *
					 (forref21[ig*4+indf_21] - forref21[ig*4+indf_21-1]));
				//ssa(lay,ngs20+ig) = tauray_21/ztaug(lay,ngs20+ig)
				ztaur[(ngs20+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_21;
			}
		}


	//******************taumol22****************

		laysolfr_22 = laytrop[iplon];
// Lower atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			if ((jp[lay*ncol+iplon] < layreffr_22) && (jp[(lay+1)*ncol+iplon] >= layreffr_22))
				laysolfr_22 = (((lay+1)<laytrop[iplon])?(lay+1):laytrop[iplon]);
			o2cont_22 = (4.35e-4)*colo2[lay*ncol+iplon]/(350.0*2);
			speccomb_22 = colh2o[lay*ncol+iplon] + o2adj_22*strrat_22*colo2[lay*ncol+iplon];
			specparm_22 = colh2o[lay*ncol+iplon]/speccomb_22;

			if (specparm_22 >= oneminus) specparm_22 = oneminus;
			 specmult_22 = 8.0*(specparm_22);
			//odadj = specparm_22 + o2adj_22 * (1.0 - specparm_22);
			 js_22 = 1 + (int)(specmult_22);
			 fs_22 =specmult_22-(int)specmult_22;
			 fac000_22 = (1.0 - fs_22) * fac00[lay*ncol+iplon];
			 fac010_22 = (1.0 - fs_22) * fac10[lay*ncol+iplon];
			 fac100_22 = fs_22 * fac00[lay*ncol+iplon];
			 fac110_22 = fs_22 * fac10[lay*ncol+iplon];
			 fac001_22 = (1.0 - fs_22) * fac01[lay*ncol+iplon];
			 fac011_22 = (1.0 - fs_22) * fac11[lay*ncol+iplon];
			 fac101_22 = fs_22 * fac01[lay*ncol+iplon];
			 fac111_22 = fs_22 * fac11[lay*ncol+iplon];
			 ind0_22 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[6] + js_22;
			 ind1_22 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[6] + js_22;
			 inds_22 = indself[lay*ncol+iplon];
			 indf_22 = indfor[lay*ncol+iplon];
			 tauray_22 = colmol[lay*ncol+iplon] * rayl_22;

			for(ig = 0;ig< ng22;ig++)
			{
				ztaug[(ngs21+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_22 *
					(fac000_22 * absa22[ig*585+ind0_22-1] +
					 fac100_22 * absa22[ig*585+ind0_22] +
					 fac010_22 * absa22[ig*585+ind0_22+8] +
					 fac110_22 * absa22[ig*585+ind0_22+9] +
					 fac001_22 * absa22[ig*585+ind1_22-1] +
					 fac101_22 * absa22[ig*585+ind1_22] +
					 fac011_22 * absa22[ig*585+ind1_22+8] +
					 fac111_22 * absa22[ig*585+ind1_22+9]) +
					 colh2o[lay*ncol+iplon] *
					 (selffac[lay*ncol+iplon] * (selfref22[ig*10+inds_22-1] +
					 selffrac[lay*ncol+iplon] *
					  (selfref22[ig*10+inds_22] - selfref22[ig*10+inds_22-1])) +
					 forfac[lay*ncol+iplon] * (forref22[ig*3+indf_22-1] +
					 forfrac[lay*ncol+iplon] *
					 (forref22[ig*3+indf_22] - forref22[ig*3+indf_22-1])))
					 + o2cont_22;
				//ssa(lay,ngs21+ig) = tauray_22/ztaug(lay,ngs21+ig)
				if (lay == laysolfr_22 - 1) zsflxzen[(ngs21+ig)*ncol+iplon] = sfluxref22[(js_22-1)*2+ig]
					+ fs_22 * (sfluxref22[js_22*2+ig] - sfluxref22[(js_22-1)*2+ig]);
				ztaur[(ngs21+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_22;
			}
		}

// Upper atmosphere loop
		  if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		  {
			 o2cont_22 = (4.35e-4)*colo2[lay*ncol+iplon]/(350.0*2);
			 ind0_22 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[6] + 1;
			 ind1_22 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[6]+ 1;
			 tauray_22 = colmol[lay*ncol+iplon] * rayl_22;

			 for(ig = 0;ig< ng22;ig++)
			 {
				ztaug[(ngs21+ig)*nlayers*ncol+lay*ncol+iplon]  = colo2[lay*ncol+iplon] * o2adj_22 *
					(fac00[lay*ncol+iplon] * absb22[ig*235+ind0_22-1] +
					 fac10[lay*ncol+iplon] * absb22[ig*235+ind0_22] +
					 fac01[lay*ncol+iplon] * absb22[ig*235+ind1_22-1] +
					 fac11[lay*ncol+iplon] * absb22[ig*235+ind1_22]) +
					 o2cont_22;
				//ssa(lay,ngs21+ig) = tauray_22/ztaug(lay,ngs21+ig)
				ztaur[(ngs21+ig)*nlayers*ncol+lay*ncol+iplon]  = tauray_22;
			 }
		 }

	//******************taumol23****************

		laysolfr_23 = laytrop[iplon];
//Lower atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			if ((jp[lay*ncol+iplon]  < layreffr_23) && (jp[(lay+1)*ncol+iplon] >= layreffr_23))
				laysolfr_23 = (((lay+1)<laytrop[iplon])?(lay+1):laytrop[iplon]);
			ind0_23 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[7] + 1;
			ind1_23 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[7] + 1;
			inds_23 = indself[lay*ncol+iplon];
			indf_23 = indfor[lay*ncol+iplon];

			for( ig = 0;ig<ng23;ig++)
			{
				tauray_23 = colmol[lay*ncol+iplon] * rayl23[ig];
				ztaug[(ngs22+ig)*nlayers*ncol+lay*ncol+iplon]= colh2o[lay*ncol+iplon] *
					(givfac_23 * (fac00[lay*ncol+iplon] * absa23[ig*65+ind0_23-1] +
					 fac10[lay*ncol+iplon] * absa23[ig*65+ind0_23] +
					 fac01[lay*ncol+iplon] * absa23[ig*65+ind1_23-1] +
					 fac11[lay*ncol+iplon] * absa23[ig*65+ind1_23]) +
					 selffac[lay*ncol+iplon] * (selfref23[ig*10+inds_23-1] +
					 selffrac[lay*ncol+iplon] *
					 (selfref23[ig*10+inds_23] - selfref23[ig*10+inds_23-1])) +
					 forfac[lay*ncol+iplon] * (forref23[ig*3+indf_23-1] +
					 forfrac[lay*ncol+iplon] *
					 (forref23[ig*3+indf_23] - forref23[ig*3+indf_23-1]))) ;
			  // ssa(lay,ngs22+ig) = tauray_23/ztaug(lay,ngs22+ig)
				if (lay == laysolfr_23 - 1) zsflxzen[(ngs22+ig)*ncol+iplon] = sfluxref23[ig];
				ztaur[(ngs22+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_23;
			 }
		}

// Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			for(ig = 0;ig<ng23;ig++)
			{
				//ztaug(lay,ngs22+ig) = colmol(lay) * rayl(ig)
				//ssa(lay,ngs22+ig) = 1.0_r8
				ztaug[(ngs22+ig)*nlayers*ncol+lay*ncol+iplon] = 0.0;
				ztaur[(ngs22+ig)*nlayers*ncol+lay*ncol+iplon] = colmol[lay*ncol+iplon] * rayl23[ig];
			}
		}

	//******************taumol24****************

		laysolfr_24 = laytrop[iplon];
	//Lower atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			 if ((jp[lay*ncol+iplon] < layreffr_24) && (jp[(lay+1)*ncol+iplon] >= layreffr_24))
				laysolfr_24 =(((lay+1)<laytrop[iplon])?(lay+1):laytrop[iplon]);
			 speccomb_24 = colh2o[lay*ncol+iplon] + strrat_24*colo2[lay*ncol+iplon];
			 specparm_24 = colh2o[lay*ncol+iplon]/speccomb_24 ;

			 if (specparm_24 >= oneminus) specparm_24 = oneminus;
			 specmult_24 = 8.0*(specparm_24);
			 js_24 = 1 + int(specmult_24);
			 fs_24 =specmult_24-(int)specmult_24;
			 fac000_24 = (1.0 - fs_24) * fac00[lay*ncol+iplon];
			 fac010_24 = (1.0 - fs_24) * fac10[lay*ncol+iplon];
			 fac100_24 = fs_24 * fac00[lay*ncol+iplon];
			 fac110_24 = fs_24 * fac10[lay*ncol+iplon];
			 fac001_24 = (1.0 - fs_24) * fac01[lay*ncol+iplon];
			 fac011_24 = (1.0 - fs_24) * fac11[lay*ncol+iplon];
			 fac101_24 = fs_24 * fac01[lay*ncol+iplon];
			 fac111_24 = fs_24 * fac11[lay*ncol+iplon];
			 ind0_24 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[8] + js_24;
			 ind1_24 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[8] + js_24;
			 inds_24 = indself[lay*ncol+iplon];
			 indf_24 = indfor[lay*ncol+iplon];

			for(ig = 0;ig< ng24;ig++)
			{
				tauray_24 = colmol[lay*ncol+iplon] * (rayla24[(js_24-1)*8+ig] +
				   fs_24 * (rayla24[js_24*8+ig] - rayla24[(js_24-1)*8+ig]));
				ztaug[(ngs23+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_24 *
					(fac000_24 * absa24[ig*585+ind0_24-1] +
					 fac100_24 * absa24[ig*585+ind0_24] +
					 fac010_24 * absa24[ig*585+ind0_24+8]+
					 fac110_24 * absa24[ig*585+ind0_24+9] +
					 fac001_24 * absa24[ig*585+ind1_24-1] +
					 fac101_24 * absa24[ig*585+ind1_24] +
					 fac011_24 * absa24[ig*585+ind1_24+8] +
					 fac111_24 * absa24[ig*585+ind1_24+9]) +
					 colo3[lay*ncol+iplon] * abso3a24[ig] +
					 colh2o[lay*ncol+iplon] *
					 (selffac[lay*ncol+iplon] * (selfref24[ig*10+inds_24-1] +
					 selffrac[lay*ncol+iplon] *
					 (selfref24[ig*10+inds_24] - selfref24[ig*10+inds_24-1])) +
					 forfac[lay*ncol+iplon] * (forref24[ig*3+indf_24-1] +
					 forfrac[lay*ncol+iplon]*
					 (forref24[ig*3+indf_24] - forref24[ig*3+indf_24-1])));
				//ssa(lay,ngs23+ig) = tauray_24/ztaug(lay,ngs23+ig)
				if (lay == laysolfr_24 - 1) zsflxzen[(ngs23+ig)*ncol+iplon] = sfluxref24[(js_24-1)*8+ig]
				   + fs_24 * (sfluxref24[js_24*8+ig] - sfluxref24[(js_24-1)*8+ig]);
				ztaur[(ngs23+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_24;
			}
		}

//Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			 ind0_24 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[8] + 1;
			 ind1_24 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[8] + 1;

			for( ig = 0;ig< ng24;ig++)
			{
				tauray_24 = colmol[lay*ncol+iplon] * raylb24[ig];
				ztaug[(ngs23+ig)*nlayers*ncol+lay*ncol+iplon] = colo2[lay*ncol+iplon] *
					(fac00[lay*ncol+iplon] * absb24[ig*235+ind0_24-1] +
					 fac10[lay*ncol+iplon] * absb24[ig*235+ind0_24] +
					 fac01[lay*ncol+iplon] * absb24[ig*235+ind1_24-1] +
					 fac11[lay*ncol+iplon] * absb24[ig*235+ind1_24]) +
					 colo3[lay*ncol+iplon] * abso3b24[ig];
				//ssa(lay,ngs23+ig) = tauray_24/ztaug(lay,ngs23+ig)
				ztaur[(ngs23+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_24;
			}
		}

	//******************taumol25****************

		laysolfr_25 = laytrop[iplon];
//Lower atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			if ((jp[lay*ncol+iplon] < layreffr_25) && (jp[(lay+1)*ncol+iplon] >= layreffr_25))
				laysolfr_25 = (((lay+1)<laytrop[iplon])?(lay+1):laytrop[iplon]);
			ind0_25 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[9] + 1;
			ind1_25 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[9] + 1;

			for(ig = 0;ig< ng25;ig++)
			{
				tauray_25 = colmol[lay*ncol+iplon] * rayl25[ig];
				ztaug[(ngs24+ig)*nlayers*ncol+lay*ncol+iplon] = colh2o[lay*ncol+iplon]*
					(fac00[lay*ncol+iplon] * absa25[ig*65+ind0_25-1] +
					 fac10[lay*ncol+iplon] * absa25[ig*65+ind0_25] +
					 fac01[lay*ncol+iplon] * absa25[ig*65+ind1_25-1] +
					 fac11[lay*ncol+iplon]* absa25[ig*65+ind1_25]) +
					 colo3[lay*ncol+iplon] * abso3a25[ig] ;
				//ssa(lay,ngs24+ig) = tauray_25/ztaug(lay,ngs24+ig)
				if (lay == laysolfr_25 - 1) zsflxzen[(ngs24+ig)*ncol+iplon] = sfluxref25[ig];
				ztaur[(ngs24+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_25;
			}
		}

	//Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			for(ig = 0;ig<ng25;ig++)
			{
				tauray_25 = colmol[lay*ncol+iplon] * rayl25[ig];
				ztaug[(ngs24+ig)*nlayers*ncol+lay*ncol+iplon] = colo3[lay*ncol+iplon] * abso3b25[ig];
				//ssa(lay,ngs24+ig) = tauray_25/ztaug(lay,ngs24+ig)
				ztaur[(ngs24+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_25;
			}
		}

	//******************taumol26****************

		laysolfr_26 = laytrop[iplon];
//Lower atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			for(ig = 0;ig<ng26;ig++)
			{
				if (lay == laysolfr_26 - 1) zsflxzen[(ngs25+ig)*ncol+iplon] = sfluxref26[ig];
				ztaug[(ngs25+ig)*nlayers*ncol+lay*ncol+iplon] = 0.0;
				ztaur[(ngs25+ig)*nlayers*ncol+lay*ncol+iplon] = colmol[lay*ncol+iplon] * rayl26[ig];
			}
		}

// Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			for(ig = 0;ig<ng26;ig++)
			{
				//ztaug(lay,ngs25+ig) = colmol(lay) * rayl(ig)
				// ssa(lay,ngs25+ig) = 1.0_r8
				ztaug[(ngs25+ig)*nlayers*ncol+lay*ncol+iplon] = 0.0;
				ztaur[(ngs25+ig)*nlayers*ncol+lay*ncol+iplon] = colmol[lay*ncol+iplon] * rayl26[ig];
			}
		}


	//******************taumol27****************

		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			ind0_27 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[11] + 1;
			ind1_27 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[11] + 1;

			for(ig = 0;ig< ng27;ig++)
			{
				tauray_27 = colmol[lay*ncol+iplon] * rayl27[ig];
				ztaug[(ngs26+ig)*nlayers*ncol+lay*ncol+iplon] = colo3[lay*ncol+iplon] *
					(fac00[lay*ncol+iplon] * absa27[ig*65+ind0_27-1] +
					fac10[lay*ncol+iplon] * absa27[ig*65+ind0_27] +
					fac01[lay*ncol+iplon] * absa27[ig*65+ind1_27-1] +
					fac11[lay*ncol+iplon] * absa27[ig*65+ind1_27]);
				//ssa(lay,ngs26+ig) = tauray_27/ztaug(lay,ngs26+ig)
				ztaur[(ngs26+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_27;
			}
		}

		laysolfr_27 = nlayers;

// Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			if ((jp[(lay-1)*ncol+iplon] < layreffr_27) && (jp[lay*ncol+iplon] >= layreffr_27))
				laysolfr_27 = lay+1;
			ind0_27 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[11] + 1;
			ind1_27 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[11] + 1;

			for(ig =0;ig<ng27;ig++)
			{
				tauray_27 = colmol[lay*ncol+iplon] * rayl27[ig];
				ztaug[(ngs26+ig)*nlayers*ncol+lay*ncol+iplon] = colo3[lay*ncol+iplon] *
					(fac00[lay*ncol+iplon] * absb27[ig*235+ind0_27-1] +
					fac10[lay*ncol+iplon] * absb27[ig*235+ind0_27] +
					fac01[lay*ncol+iplon] * absb27[ig*235+ind1_27-1] +
					fac11[lay*ncol+iplon] * absb27[ig*235+ind1_27]);
				//ssa(lay,ngs26+ig) = tauray_27/ztaug(lay,ngs26+ig)
				if (lay == laysolfr_27 - 1) zsflxzen[(ngs26+ig)*ncol+iplon] = scalekur_27 * sfluxref27[ig];
				ztaur[(ngs26+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_27;
			}
		}

	//******************taumol28****************

		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			speccomb_28 = colo3[lay*ncol+iplon] + strrat_28*colo2[lay*ncol+iplon];
			specparm_28 = colo3[lay*ncol+iplon]/speccomb_28;
			if (specparm_28 >= oneminus) specparm_28 = oneminus;
			specmult_28 = 8.0*(specparm_28);
			js_28 = 1 + (int)(specmult_28);
			fs_28 =specmult_28 - (int)specmult_28;
			fac000_28 = (1.0 - fs_28) * fac00[lay*ncol+iplon];
			fac010_28 = (1.0 - fs_28) * fac10[lay*ncol+iplon];
			fac100_28 = fs_28 * fac00[lay*ncol+iplon];
			fac110_28 = fs_28 * fac10[lay*ncol+iplon];
			fac001_28 = (1.0 - fs_28) * fac01[lay*ncol+iplon];
			fac011_28 = (1.0 - fs_28) * fac11[lay*ncol+iplon];
			fac101_28 = fs_28 * fac01[lay*ncol+iplon];
			fac111_28 = fs_28 * fac11[lay*ncol+iplon];
			ind0_28 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[12] + js_28;
			ind1_28 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[12] + js_28;
			tauray_28 = colmol[lay*ncol+iplon] * rayl_28;

			for(ig = 0;ig< ng28;ig++)
			{
				ztaug[(ngs27+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_28 *
					(fac000_28 * absa28[ig*585+ind0_28-1] +
					fac100_28 * absa28[ig*585+ind0_28] +
					fac010_28 * absa28[ig*585+ind0_28+8] +
					fac110_28 * absa28[ig*585+ind0_28+9] +
					fac001_28 * absa28[ig*585+ind1_28-1] +
					fac101_28 * absa28[ig*585+ind1_28] +
					fac011_28 * absa28[ig*585+ind1_28+8] +
					fac111_28 * absa28[ig*585+ind1_28+9]) ;
				//ssa(lay,ngs27+ig) = tauray_28/ztaug(lay,ngs27+ig)
				ztaur[(ngs27+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_28;
			}
		}

		laysolfr_28 = nlayers;

// Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			if ((jp[(lay-1)*ncol+iplon] < layreffr_28) && (jp[lay*ncol+iplon] >= layreffr_28) )
				laysolfr_28 = lay+1;
			speccomb_28 = colo3[lay*ncol+iplon] + strrat_28*colo2[lay*ncol+iplon];
			specparm_28 = colo3[lay*ncol+iplon]/speccomb_28 ;
			if (specparm_28 >= oneminus) specparm_28 = oneminus;
			specmult_28 = 4.0*(specparm_28);
			js_28 = 1 + (int)(specmult_28);
			fs_28 =specmult_28 - (int)specmult_28;
			fac000_28 = (1.0 - fs_28) * fac00[lay*ncol+iplon];
			fac010_28 = (1.0 - fs_28) * fac10[lay*ncol+iplon];
			fac100_28 = fs_28 * fac00[lay*ncol+iplon];
			fac110_28 = fs_28 * fac10[lay*ncol+iplon];
			fac001_28 = (1.0 - fs_28) * fac01[lay*ncol+iplon];
			fac011_28 = (1.0 - fs_28) * fac11[lay*ncol+iplon];
			fac101_28 = fs_28 * fac01[lay*ncol+iplon];
			fac111_28 = fs_28 * fac11[lay*ncol+iplon];
			ind0_28 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[12] + js_28;
			ind1_28 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[12] + js_28;
			tauray_28 = colmol[lay*ncol+iplon]* rayl_28;

			for(ig = 0;ig<ng28;ig++)
			{
				ztaug[(ngs27+ig)*nlayers*ncol+lay*ncol+iplon] = speccomb_28 *
					(fac000_28 * absb28[ig*1175+ind0_28-1] +
					fac100_28 * absb28[ig*1175+ind0_28] +
					fac010_28 * absb28[ig*1175+ind0_28+4] +
					fac110_28 * absb28[ig*1175+ind0_28+5] +
					fac001_28 * absb28[ig*1175+ind1_28-1] +
					fac101_28 * absb28[ig*1175+ind1_28] +
					fac011_28 * absb28[ig*1175+ind1_28+4] +
					fac111_28 * absb28[ig*1175+ind1_28+5]) ;
				//ssa(lay,ngs27+ig) = tauray_28/ztaug(lay,ngs27+ig)
				if (lay == laysolfr_28 - 1) zsflxzen[(ngs27+ig)*ncol+iplon] = sfluxref28[(js_28-1)*6+ig]
					+ fs_28 * (sfluxref28[js_28*6+ig] - sfluxref28[(js_28-1)*6+ig]);
				ztaur[(ngs27+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_28;
			}
		}

	//******************taumol29****************
		rayl_29=1.0;

		if(iplon>=0&&iplon<ncol&&lay>=0&&lay<laytrop[iplon])
		{
			 ind0_29 = ((jp[lay*ncol+iplon]-1)*5+(jt[lay*ncol+iplon]-1))*nspa[13] + 1;
			 ind1_29 = (jp[lay*ncol+iplon]*5+(jt1[lay*ncol+iplon]-1))*nspa[13] + 1;
			 inds_29 = indself[lay*ncol+iplon];
			 indf_29 = indfor[lay*ncol+iplon];
			 tauray_29 = colmol[lay*ncol+iplon] * rayl_29;

			for(ig = 0;ig< ng29;ig++)
			{
				ztaug[(ngs28+ig)*nlayers*ncol+lay*ncol+iplon] = colh2o[lay*ncol+iplon] *
					((fac00[lay*ncol+iplon] * absa29[ig*65+ind0_29 - 1] +
					fac10[lay*ncol+iplon] * absa29[ig*65+ind0_29] +
					fac01[lay*ncol+iplon] * absa29[ig*65+ind1_29 - 1] +
					fac11[lay*ncol+iplon] * absa29[ig*65+ind1_29]) +
					selffac[lay*ncol+iplon] * (selfref29[ig*10+inds_29 - 1] +
					selffrac[lay*ncol+iplon] *
					(selfref29[ig*10+inds_29] - selfref29[ig*10+inds_29 -1])) +
					forfac[lay*ncol+iplon]* (forref29[ig*4+indf_29 - 1] +
					forfrac[lay*ncol+iplon] *
					(forref29[ig*4+indf_29] - forref29[ig*4+indf_29 - 1])))
					+ colco2[lay*ncol+iplon] * absco229[ig];

				//ssa(lay,ngs28+ig) = tauray_29/ztaug(lay,ngs28+ig)
				ztaur[(ngs28+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_29;
			}
		}

		laysolfr_29 = nlayers;

// Upper atmosphere loop
		if(iplon>=0&&iplon<ncol&&lay >= laytrop[iplon]&&lay<nlayers)
		{
			if ((jp[(lay-1)*ncol+iplon] < layreffr_29) && (jp[lay*ncol+iplon] >= layreffr_29) )
				laysolfr_29 = lay+1;
			ind0_29 = ((jp[lay*ncol+iplon]-13)*5+(jt[lay*ncol+iplon]-1))*nspb[13] + 1;
			ind1_29 = ((jp[lay*ncol+iplon]-12)*5+(jt1[lay*ncol+iplon]-1))*nspb[13] + 1;
			tauray_29 = colmol[lay*ncol+iplon] * rayl_29;

			for(ig = 0;ig<ng29;ig++)
			{
				ztaug[(ngs28+ig)*nlayers*ncol+lay*ncol+iplon] = colco2[lay*ncol+iplon] *
					(fac00[lay*ncol+iplon] * absb29[ig*235+ind0_29-1] +
					fac10[lay*ncol+iplon] * absb29[ig*235+ind0_29] +
					fac01[lay*ncol+iplon] * absb29[ig*235+ind1_29-1] +
					fac11[lay*ncol+iplon] * absb29[ig*235+ind1_29])
					+ colh2o[lay*ncol+iplon] * absh2o29[ig] ;
				//ssa(lay,ngs28+ig) = tauray_29/ztaug(lay,ngs28+ig)
				if (lay == laysolfr_29 - 1) zsflxzen[(ngs28+ig)*ncol+iplon]=sfluxref29[ig];
				ztaur[(ngs28+ig)*nlayers*ncol+lay*ncol+iplon] = tauray_29;
			}
		}

            return;
}

__device__ void  reftra_sw(double bpade,double *exp_tbl,
							int *lrtchkclr,double *zgcc,
							double *prmu0_d,double *ztauc,double *zomcc,
							double *ztrac,double *ztradc,double *zrefc,double *zrefdc,
							int iplon)
{
// ----- Local -----

// -------reftra Local -------
      int kmodts,jk;

	  double tblind;
	  int itind;

      double za, za1, za2;
      double zbeta, zdend, zdenr, zdent;
      double ze1,ze2, zem1, zem2, zemm, zep1, zep2;
      double zg, zg3, zgamma1, zgamma2, zgamma3, zgamma4, zgt;
      double zr1, zr2, zr3, zr4, zr5;
      double zrk, zrk2, zrkg, zrm1, zrp, zrp1, zrpp;
      double zsr3, zt1, zt2, zt3, zt4, zt5, zto1;
      double zw, zwcrit, zwo;

	  const double eps = 1.e-08;

    //reftra  Initializations
		zsr3=sqrt(3.0);
		zwcrit=0.9999995;
		kmodts=2;

     //reftra计算部分
		for(jk=0;jk< nlayers;jk++)
		{
			if (!lrtchkclr[jk*ncol+iplon])
			{
				zrefc[jk*ncol+iplon] =0.0;
				ztrac[jk*ncol+iplon] =1.0;
				zrefdc[jk*ncol+iplon]=0.0;
				ztradc[jk*ncol+iplon]=1.0;
			}
			else
			{
				zto1=ztauc[jk*ncol+iplon];
				zw  =zomcc[jk*ncol+iplon];
				zg  =zgcc[jk*ncol+iplon];
				zg3= 3.0 * zg;

				if (kmodts == 1)
				{
					zgamma1= (7.0 - zw * (4.0 + zg3)) * 0.25;
					zgamma2=-(1.0 - zw * (4.0 - zg3)) * 0.25;
					zgamma3= (2.0 - zg3 * prmu0_d[iplon] ) * 0.25;
				}
				else if (kmodts == 2)
				{
					zgamma1=(8.0 - zw * (5.0 + zg3)) * 0.25;
					zgamma2=  3.0 *(zw * (1.0 - zg )) * 0.25;
					zgamma3= (2.0 - zg3 * prmu0_d[iplon] ) * 0.25;
				}
				else if (kmodts == 3)
				{
					zgamma1= zsr3 * (2.0 - zw * (1.0 + zg)) * 0.5;
					zgamma2= zsr3 * zw * (1.0 - zg ) * 0.5;
					zgamma3= (1.0 - zsr3 * zg * prmu0_d[iplon] ) * 0.5;
				}

				zgamma4= 1.0 - zgamma3;
				zwo= zw / (1.0 - (1.0 - zw) * ((zg / (1.0 - zg))*(zg / (1.0 - zg))));

				if (zwo >= zwcrit)
				{
//Conservative scattering
					za  = zgamma1 * prmu0_d[iplon];
					za1 = za - zgamma3;
					zgt = zgamma1 * zto1;

// Homogeneous reflectance and transmittance,
// collimated beam

					ze1 = ((( zto1 / prmu0_d[iplon])< 500.0 )? ( zto1 / prmu0_d[iplon]):500.0);
				//ze2 = exp( -ze1 );

// Use exponential lookup table for transmittance, or expansion of
// exponential for low tau
					if (ze1 <= od_lo)
						ze2 = 1.0 - ze1 + 0.5 * ze1 * ze1;
					else
					{
						tblind = ze1 / (bpade + ze1);
						itind = tblint * tblind + 0.5;
						//itind=10;
						ze2 = exp_tbl[itind];
					}

					zrefc[jk*ncol+iplon] = (zgt - za1 * (1.0 - ze2)) / (1.0 + zgt);
					ztrac[jk*ncol+iplon] = 1.0 - zrefc[jk*ncol+iplon];
					zrefdc[jk*ncol+iplon] = zgt / (1.0 + zgt);
					ztradc[jk*ncol+iplon] = 1.0 - zrefdc[jk*ncol+iplon];

// This is applied for consistency between total (delta-scaled) and direct (unscaled)
// calculations at very low optical depths (tau < 1.e-4) when the exponential lookup
// table returns a transmittance of 1.0.
					if (ze2 == 1.0)
					{
						zrefc[jk*ncol+iplon] = 0.0;
						ztrac[jk*ncol+iplon]= 1.0;
						zrefdc[jk*ncol+iplon] = 0.0;
						ztradc[jk*ncol+iplon] = 1.0;
					}
				}

// Non-conservative scattering
				else
				{
					za1 = zgamma1 * zgamma4 + zgamma2 * zgamma3;
					za2 = zgamma1 * zgamma3 + zgamma2 * zgamma4;
					zrk = sqrt( zgamma1*zgamma1 - zgamma2*zgamma2);

					zrp = zrk * prmu0_d[iplon];
					zrp1 = 1.0 + zrp;
					zrm1 = 1.0 - zrp;
					zrk2 = 2.0 * zrk;
					zrpp = 1.0 - zrp*zrp;
					zrkg = zrk + zgamma1;
					zr1  = zrm1 * (za2 + zrk * zgamma3);
					zr2  = zrp1 * (za2 - zrk * zgamma3);
					zr3  = zrk2 * (zgamma3 - za2 * prmu0_d[iplon] );
					zr4  = zrpp * zrkg;
					zr5  = zrpp * (zrk - zgamma1);
					zt1  = zrp1 * (za1 + zrk * zgamma4);
					zt2  = zrm1 * (za1 - zrk * zgamma4);
					zt3  = zrk2 * (zgamma4 + za1 * prmu0_d[iplon] );
					zt4  = zr4;
					zt5  = zr5;
					zbeta = (zgamma1 - zrk) / zrkg ;

// Homogeneous reflectance and transmittance

					ze1 =((( zrk * zto1)< 500.0)?( zrk * zto1):500.0);
					ze2 =((( zto1 / prmu0_d[iplon])<500.0)?( zto1 / prmu0_d[iplon]):500.0);
// exponential for low tau
					if (ze1 <= od_lo)
					{
						zem1 = 1.0 - ze1 + 0.5 * ze1 * ze1;
						zep1 = 1.0 / zem1;
					}
					else
					{
						tblind = ze1 / (bpade + ze1);
						itind = tblint * tblind + 0.5;
						//itind=10;
						zem1 = exp_tbl[itind];
						zep1 = 1.0 / zem1;
					}

					if (ze2 <= od_lo)
					{
						zem2 = 1.0 - ze2 + 0.5 * ze2 * ze2;
						zep2 = 1.0 / zem2;
					}
					else
					{
						tblind = ze2 / (bpade + ze2);
						itind = tblint * tblind + 0.5;
						//itind=10;
						zem2 = exp_tbl[itind];
						zep2 = 1.0 / zem2;
					}

// collimated beam
					zdenr = zr4*zep1 + zr5*zem1;
					zdent = zt4*zep1 + zt5*zem1;

					if ((zdenr >= (-eps)) && (zdenr <= eps))
					{
						zrefc[jk*ncol+iplon]= eps;
						ztrac[jk*ncol+iplon] = zem2;
					}
					else
					{
						zrefc[jk*ncol+iplon] =zw * (zr1*zep1 - zr2*zem1 - zr3*zem2) / zdenr;
						ztrac[jk*ncol+iplon] =zem2 - zem2 * zw * (zt1*zep1 - zt2*zem1 - zt3*zep2) / zdent;
					}

					zemm = zem1*zem1;
					zdend = 1.0 / ( (1.0 - zbeta*zemm ) * zrkg);
					zrefdc[jk*ncol+iplon] = zgamma2 * (1.0 - zemm) * zdend;

					ztradc[jk*ncol+iplon] =  zrk2*zem1*zdend;

				}
			}
		}
}

__device__ void vrtqdr_sw(int kw,int klev,double *pref,double *prefd,
						double *ptra,double *ptrad,double *pdbt,double *prdnd,
						double *prup,double *prupd,double *ptdbt,
						double *pfu,double *pfd,int iplon)
{

	int ikp,ikx,jk;
	double zreflect;
	double ztdn[53];

	zreflect = 1.0 / (1.0 - prefd[klev*ncol+iplon] * prefd[(klev-1)*ncol+iplon]);
	prup[(klev-1)*ncol+iplon]= pref[(klev-1)*ncol+iplon] + (ptrad[(klev-1)*ncol+iplon] *
			((ptra[(klev-1)*ncol+iplon] - pdbt[(klev-1)*ncol+iplon]) * prefd[klev*ncol+iplon] +
			pdbt[(klev-1)*ncol+iplon]* pref[klev*ncol+iplon])) * zreflect;
	prupd[(klev-1)*ncol+iplon] = prefd[(klev-1)*ncol+iplon] + ptrad[(klev-1)*ncol+iplon] * ptrad[(klev-1)*ncol+iplon] *
						prefd[klev*ncol+iplon]* zreflect;

	for(jk = 0;jk<klev-1;jk++)
	{
		ikp = klev-jk-1;
		ikx = ikp-1;
		zreflect= 1.0 / (1.0 -prupd[ikp*ncol+iplon] * prefd[ikx*ncol+iplon]);
		prup[ikx*ncol+iplon]= pref[ikx*ncol+iplon] + (ptrad[ikx*ncol+iplon] *
				((ptra[ikx*ncol+iplon] - pdbt[ikx*ncol+iplon]) * prupd[ikp*ncol+iplon] +
					pdbt[ikx*ncol+iplon] * prup[ikp*ncol+iplon])) * zreflect;
		prupd[ikx*ncol+iplon]= prefd[ikx*ncol+iplon] + ptrad[ikx*ncol+iplon] * ptrad[ikx*ncol+iplon] *
							prupd[ikp*ncol+iplon] * zreflect;
	}

	ztdn[0]=1.0;
	prdnd[0*ncol+iplon]=0.0;
	ztdn[1]=ptra[0*ncol+iplon];
	prdnd[1*ncol+iplon]=prefd[0*ncol+iplon];

	for(jk = 1;jk<klev;jk++)
	{
		ikp = jk+1;
		zreflect = 1.0 / (1.0 - prefd[jk*ncol+iplon] * prdnd[jk*ncol+iplon]);
		ztdn[ikp] = ptdbt[jk*ncol+iplon] * ptra[jk*ncol+iplon] +
					(ptrad[jk*ncol+iplon] * ((ztdn[jk]- ptdbt[jk*ncol+iplon]) +
					ptdbt[jk*ncol+iplon] * pref[jk*ncol+iplon] * prdnd[jk*ncol+iplon])) * zreflect;
		prdnd[ikp*ncol+iplon] =prefd[jk*ncol+iplon] + ptrad[jk*ncol+iplon] * ptrad[jk*ncol+iplon] *
					prdnd[jk*ncol+iplon] * zreflect;

	}
	for( jk = 0;jk<klev+1;jk++)
	{
		zreflect = 1.0 / (1.0 - prdnd[jk*ncol+iplon] * prupd[jk*ncol+iplon]);
		pfu[kw*(klev+1)*ncol+jk*ncol+iplon] = (ptdbt[jk*ncol+iplon] * prup[jk*ncol+iplon] +
		(ztdn[jk] - ptdbt[jk*ncol+iplon]) * prupd[jk*ncol+iplon]) * zreflect;

		pfd[kw*(klev+1)*ncol+jk*ncol+iplon] =ptdbt[jk*ncol+iplon] + (ztdn[jk] - ptdbt[jk*ncol+iplon]+
				ptdbt[jk*ncol+iplon] * prup[jk*ncol+iplon]* prdnd[jk*ncol+iplon])* zreflect;
	}

}
__global__ void spcvmc_sw1(int istart,int iend,double *pbbcd,double *pbbcu,
	double *pbbfd,double *pbbfu,double *pbbcddir,double *pbbfddir,
	double *puvcd,double *puvfd,double *puvcddir,double *puvfddir,
	double *pnicd,double *pnifd,double *pnicddir,double *pnifddir,
	double *pnicu,double *pnifu,double *zsflxzen,double *ztaug,double *ztaur,
	int *ngc,int *ngs,double bpade,double *exp_tbl,int icpr,int idelm,
	int iout,double *ptaucmc_d,int *lrtchkclr,double *zgcc,double *prmu0_d,
	double *ztauo,double *ztauc,double *zomcc,double *ztrac,double *ztradc,
	double *zrefc,double *zrefdc,double *zincflx,double *adjflux,double *ztdbtc,
	double *ztdbtc_nodel,double *zdbtc,double *palbp_d,double *palbd_d,
	double *ztdbt,double *ztdbt_nodel,double *zdbt,double *ztra,double *ztrad,
	double *zref,double *zrefd,int *lrtchkcld,double *pcldfmc_d,
	double *ptaua_d,double *pomga_d,double *pasya_d,double *ptaormc_d,double *zomco,
	double *pomgcmc_d,double *zgco,double *pasycmc_d,double *zdbtc_nodel,
	double *zrdndc,double *zrupc,double *zrupdc,double *zcu,double *zcd,
	double *zrdnd,double *zrup,double *zrupd,double *zfu,double *zfd,double *ztrao,
	double *ztrado,double *zrefo,double *zrefdo, int offset)
	{
	int ib1,ib2,ibm,igt,ikl,ikp,ikx,jb,jg,jl,jk,klev,iw;
	int iplon,lay;
	int itind,i,j,k;
	double tblind,ze1;
    double zclear,zcloud;
    double zdbt_nodel[53];
    double zgc[52];
    double zomc[52];
    double zs1[53];
    double ztdn[53],ztdnd[53];
    double ztoc[52],ztor[52];

    double zdbtmc,zdbtmo,zf,zgw;
    double zwf,tauorig,repclc;

	iplon=blockIdx.x * blockDim.x + threadIdx.x;
    jb=blockIdx.y * blockDim.y + threadIdx.y;
	
    if(iplon>=offset && iplon<offset + ncol/4&&jb>=15&&jb<29)
	{
		klev = nlayers;
        iw = -1;
        repclc = 1.e-12;	
		   ibm = jb-15;
		   igt = ngc[ibm];
		   for( i=0;i<jb-15;++i)
		  	iw = iw + ngc[i];
			  for(jg = 0;jg<igt;jg++)
			  {	 
				  iw++;
				  zincflx[iw*ncol+iplon]= adjflux[jb*ncol+iplon]*zsflxzen[iw*ncol+iplon]  * prmu0_d[iplon];
  
				  ztdbtc[0*ncol+iplon]=1.0;
				  ztdbtc_nodel[0*ncol+iplon]=1.0;
				  zdbtc[klev*ncol+iplon] =0.0;
				  ztrac[klev*ncol+iplon] =0.0;
				  ztradc[klev*ncol+iplon]=0.0;
				  zrefc[klev*ncol+iplon] =palbp_d[ibm*ncol+iplon];
				  zrefdc[klev*ncol+iplon]=palbd_d[ibm*ncol+iplon];
				  zrupc[klev*ncol+iplon] =palbp_d[ibm*ncol+iplon];
				  zrupdc[klev*ncol+iplon]=palbd_d[ibm*ncol+iplon];
  
				  ztrao[klev*ncol+iplon]=0.0;
				  ztrado[klev*ncol+iplon]=0.0;
				  zrefo[klev*ncol+iplon] =palbp_d[ibm*ncol+iplon];
				  zrefdo[klev*ncol+iplon]=palbd_d[ibm*ncol+iplon];
  
				  ztdbt[0*ncol+iplon]=1.0;
				  ztdbt_nodel[0*ncol+iplon]=1.0;
				  zdbt[klev*ncol+iplon] =0.0;
				  ztra[klev*ncol+iplon] =0.0;
				  ztrad[klev*ncol+iplon]=0.0;
				  zref[klev*ncol+iplon] =palbp_d[ibm*ncol+iplon];
				  zrefd[klev*ncol+iplon]=palbd_d[ibm*ncol+iplon];
				  zrup[klev*ncol+iplon] =palbp_d[ibm*ncol+iplon];
				  zrupd[klev*ncol+iplon]=palbd_d[ibm*ncol+iplon];
				  for(jk=0;jk<klev;jk++)
				  {
					  ikl=klev-jk-1;
  
					  lrtchkclr[jk*ncol+iplon]=1;
					  lrtchkcld[jk*ncol+iplon]=0;
					  lrtchkcld[jk*ncol+iplon]=(pcldfmc_d[iw*nlayers+ikl] > repclc);
  
					  ztauc[jk*ncol+iplon] =ztaur[iw*nlayers*ncol+ikl*ncol+iplon] +
							  ztaug[iw*nlayers*ncol+ikl*ncol+iplon] + ptaua_d[ibm*nlayers*ncol+ikl*ncol+iplon];
  
					  zomcc[jk*ncol+iplon] = ztaur[iw*nlayers*ncol+ikl*ncol+iplon]* 1.0 + ptaua_d[ibm*nlayers*ncol+ikl*ncol+iplon] * pomga_d[ibm*nlayers*ncol+ikl*ncol+iplon];
					  zgcc[jk*ncol+iplon] = pasya_d[ibm*nlayers*ncol+ikl*ncol+iplon] * pomga_d[ibm*nlayers*ncol+ikl*ncol+iplon] * ptaua_d[ibm*nlayers*ncol+ikl*ncol+iplon] / zomcc[jk*ncol+iplon];
					  zomcc[jk*ncol+iplon]= zomcc[jk*ncol+iplon] / ztauc[jk*ncol+iplon];
  
  
					  if (idelm == 0)
					  {
						  zclear = 1.0 - pcldfmc_d[iw*nlayers+ikl];
						  zcloud = pcldfmc_d[iw*nlayers+ikl];
  
						  ze1 = ztauc[jk*ncol+iplon]/ prmu0_d[iplon];
  
						  if (ze1 <= od_lo)
							  zdbtmc = 1.0 - ze1 + 0.5 * ze1 * ze1;
						  else
						  {
							  tblind = ze1 / (bpade + ze1);
							  itind = tblint* tblind + 0.5;
							  //itind=10;
							  zdbtmc =exp_tbl[itind];
						  }
  
						  zdbtc_nodel[jk*ncol+iplon] = zdbtmc;
						  ztdbtc_nodel[(jk+1)*ncol+iplon] = zdbtc_nodel[jk*ncol+iplon] * ztdbtc_nodel[jk*ncol+iplon];
						  tauorig = ztauc[jk*ncol+iplon] + ptaormc_d[iw*nlayers+ikl];
						  ze1 = tauorig / prmu0_d[iplon];
  
						  if (ze1 <= od_lo)
							  zdbtmo = 1.0 - ze1 + 0.5 * ze1 * ze1;
						  else
						  {
							  tblind = ze1 / (bpade + ze1);
							  itind = tblint * tblind + 0.5;
							  //itind=10;
							  zdbtmo = exp_tbl[itind];
  
						  }
  
						  zdbt_nodel[jk] = zclear*zdbtmc + zcloud*zdbtmo;
						  ztdbt_nodel[(jk+1)*ncol+iplon] = zdbt_nodel[jk] * ztdbt_nodel[jk*ncol+iplon];
  
					  }

					  zf = zgcc[jk*ncol+iplon]* zgcc[jk*ncol+iplon];
					  zwf = zomcc[jk*ncol+iplon] * zf;
					  ztauc[jk*ncol+iplon] = (1.0 - zwf) * ztauc[jk*ncol+iplon];
					  zomcc[jk*ncol+iplon] = (zomcc[jk*ncol+iplon]- zwf) / (1.0 - zwf);
					  zgcc [jk*ncol+iplon] = (zgcc[jk*ncol+iplon] - zf) / (1.0 - zf);
  
					  if (icpr >= 1)
					  {
						  ztauo[jk*ncol+iplon] = ztauc[jk*ncol+iplon] + ptaucmc_d[iw*nlayers+ikl];
						  zomco[jk*ncol+iplon] = ztauc[jk*ncol+iplon] * zomcc[jk*ncol+iplon] + ptaucmc_d[iw*nlayers+ikl] * pomgcmc_d[iw*nlayers+ikl];
						  zgco[jk*ncol+iplon] = (ptaucmc_d[iw*nlayers+ikl] * pomgcmc_d[iw*nlayers+ikl] * pasycmc_d[iw*nlayers+ikl] +
								ztauc[jk*ncol+iplon] * zomcc[jk*ncol+iplon] * zgcc[jk*ncol+iplon]) / zomco[jk*ncol+iplon];
						  zomco[jk*ncol+iplon] = zomco[jk*ncol+iplon] / ztauo[jk*ncol+iplon];
					  }
					  else if (icpr == 0)
					  {
						  ztauo[jk*ncol+iplon] = ztaur[iw*nlayers*ncol+ikl*ncol+iplon] + ztaug[iw*nlayers*ncol+ikl*ncol+iplon] + ptaua_d[ibm*nlayers*ncol+ikl*ncol+iplon] + ptaucmc_d[iw*nlayers+ikl];
						  zomco[jk*ncol+iplon] = ptaua_d[ibm*nlayers*ncol+ikl*ncol+iplon] * pomga_d[ibm*nlayers*ncol+ikl*ncol+iplon] + ptaucmc_d[iw*nlayers+ikl] * pomgcmc_d[iw*nlayers+ikl] +
								ztaur[iw*nlayers*ncol+ikl*ncol+iplon] * 1.0;
						  zgco[jk*ncol+iplon] = (ptaucmc_d[iw*nlayers+ikl] * pomgcmc_d[iw*nlayers+ikl] * pasycmc_d[iw*nlayers+ikl] +
								ptaua_d[ibm*nlayers*ncol+ikl*ncol+iplon]*pomga_d[ibm*nlayers*ncol+ikl*ncol+iplon]*pasya_d[ibm*nlayers*ncol+ikl*ncol+iplon]) / zomco[jk*ncol+iplon];
						  zomco[jk*ncol+iplon] = zomco[jk*ncol+iplon]/ ztauo[jk*ncol+iplon];

  //   Use only if subroutine rrtmg_sw_cldprop is not used to get cloud properties and to apply delta scaling
						  zf = zgco[jk*ncol+iplon] * zgco[jk*ncol+iplon];
						  zwf = zomco[jk*ncol+iplon] * zf;
						  ztauo[jk*ncol+iplon]= (1.0 - zwf) * ztauo[jk*ncol+iplon];
						  zomco[jk*ncol+iplon]= (zomco[jk*ncol+iplon] - zwf) / (1.0 - zwf);
						  zgco[jk*ncol+iplon]= (zgco[jk*ncol+iplon] - zf) / (1.0 - zf);
					  }
				  }
  
				  reftra_sw(bpade,exp_tbl,lrtchkclr,zgcc,
							  prmu0_d,ztauc,zomcc,
							  ztrac,ztradc,zrefc,zrefdc,iplon);
  
						  
				  reftra_sw(bpade,exp_tbl,lrtchkcld,zgco,
							  prmu0_d,ztauo,zomco,
							  ztrao,ztrado,zrefo,zrefdo,iplon);
  
							  
				  for( jk=0;jk<klev;jk++)
				  {
					  ikl = klev-jk-1;
					  zclear = 1.0 - pcldfmc_d[iw*nlayers+ikl];
					  zcloud =pcldfmc_d[iw*nlayers+ikl];
					  zref[jk*ncol+iplon] = zclear*zrefc[jk*ncol+iplon]+ zcloud*zrefo[jk*ncol+iplon];
  
					  zrefd[jk*ncol+iplon]= zclear*zrefdc[jk*ncol+iplon]+ zcloud*zrefdo[jk*ncol+iplon];
  
					  ztra[jk*ncol+iplon] = zclear*ztrac[jk*ncol+iplon] + zcloud*ztrao[jk*ncol+iplon];
  
					  ztrad[jk*ncol+iplon]= zclear*ztradc[jk*ncol+iplon] + zcloud*ztrado[jk*ncol+iplon];
  
					  ze1 = ztauc[jk*ncol+iplon] / prmu0_d[iplon];
  
  
  
					  if (ze1 <= od_lo)
						  zdbtmc = 1.0  - ze1 + 0.5 * ze1 * ze1;
					  else
					  {
						  tblind = ze1 / (bpade + ze1);
						  itind = tblint* tblind + 0.5;
  
						  zdbtmc =exp_tbl[itind];
  
  
  
						  /**************************/
					  //	printf("zdbtmc=%E\n",zdbtmc);
						  /**************************/
					  }
  
					  zdbtc[jk*ncol+iplon] = zdbtmc;
					  ztdbtc[(jk+1)*ncol+iplon] = zdbtc[jk*ncol+iplon]*ztdbtc[jk*ncol+iplon];
  
					  /********************************/
					  // printf("zdbtc=%E\n",zdbtc[jk*ncol+iplon]);
					  /*******************************/
					  ze1 = ztauo[jk*ncol+iplon] / prmu0_d[iplon];
  
					  if (ze1 <= od_lo)
						  zdbtmo = 1.0 - ze1 + 0.5 * ze1 * ze1;
					  else
					  {
						  tblind = ze1 / (bpade + ze1);
						  itind =tblint * tblind + 0.5;
						  //itind=10;
						  zdbtmo = exp_tbl[itind];
					  }
  
					  zdbt[jk*ncol+iplon] = zclear*zdbtmc + zcloud*zdbtmo;
  
					  ztdbt[(jk+1)*ncol+iplon] = zdbt[jk*ncol+iplon]*ztdbt[jk*ncol+iplon];
  
				  }
				}
			}
	}

__global__ void spcvmc_sw2(int istart,int iend,double *pbbcd,double *pbbcu,
						double *pbbfd,double *pbbfu,double *pbbcddir,double *pbbfddir,
						double *puvcd,double *puvfd,double *puvcddir,double *puvfddir,
						double *pnicd,double *pnifd,double *pnicddir,double *pnifddir,
						double *pnicu,double *pnifu,
						int *ngc,int *ngs,int idelm,
						int iout,
						double *ztrac,double *ztradc,
						double *zrefc,double *zrefdc,double *zincflx,double *ztdbtc,
						double *ztdbtc_nodel,double *zdbtc,
						double *ztdbt,double *ztdbt_nodel,double *zdbt,double *ztra,double *ztrad,
						double *zref,double *zrefd,
						double *zrdndc,double *zrupc,double *zrupdc,double *zcu,double *zcd,
						double *zrdnd,double *zrup,double *zrupd,double *zfu,double *zfd,int offset
						)
{ 
	int ib1,ib2,ibm,igt,ikl,ikp,ikx,jb,jg,jl,jk,klev,iw;
	int iplon,lay;
	int itind,i,j,k;
	iplon=blockIdx.x * blockDim.x + threadIdx.x;
	 double pbbfu_[(nlayers+1)];
	 double pbbfd_[(nlayers+1)];
	 double pbbcu_[(nlayers+1)];
	 double pbbcd_[(nlayers+1)];
	 double pbbfddir_[(nlayers+1)];
	 double pbbcddir_[(nlayers+1)];

    if(iplon>=offset && iplon<offset + ncol/4)
	{
		klev = nlayers;
        iw = -1;
      //  repclc = 1.e-12;
        ib1 = istart;//16
        ib2 = iend;//29
		for(jk=0;jk<klev+1;jk++)
		{
			pbbcd[jk*ncol+iplon]=0;
			pbbcd_[jk]=0;
			pbbcu[jk*ncol+iplon]=0;
			pbbcu_[jk]=0;
			pbbfd[jk*ncol+iplon]=0;
			pbbfd_[jk]=0;
			pbbfu[jk*ncol+iplon]=0;
			pbbfu_[jk]=0;
			pbbcddir[jk*ncol+iplon]=0;
			pbbcddir_[jk]=0;
			pbbfddir[jk*ncol+iplon]=0;
			pbbfddir_[jk]=0;
			puvcd[jk*ncol+iplon]=0;
			//puvcd_[jk]=0;
			puvfd[jk*ncol+iplon]=0;
			//puvfd_[jk]=0;
			puvcddir[jk*ncol+iplon]=0;
			puvfddir[jk*ncol+iplon]=0;
			pnicd[jk*ncol+iplon]=0;
			pnifd[jk*ncol+iplon]=0;
			pnicddir[jk*ncol+iplon]=0;
			pnifddir[jk*ncol+iplon]=0;
			pnicu[jk*ncol+iplon]=0;
			pnifu[jk*ncol+iplon]=0;
		}
	 for(jb = ib1-1;jb< ib2;jb++)
 	 {
		ibm = jb-15;
        igt = ngc[ibm];
             	
             if (iout>0 && ibm>=1) iw = ngs[ibm-1]-1;

            for(jg = 0;jg<igt;jg++)
			{	 
            	iw++;
			
				vrtqdr_sw(iw,klev,zrefc,zrefdc,ztrac,ztradc,
					zdbtc,zrdndc,zrupc,zrupdc,ztdbtc,
					zcu,zcd,iplon);

				vrtqdr_sw(iw,klev,zref,zrefd,ztra,ztrad,
					zdbt,zrdnd,zrup,zrupd,ztdbt,
					zfu,zfd,iplon);
				
				for(jk=0;jk<klev+1;jk++)
				{
                    ikl=klev-jk;
                  
                    pbbfu_[ikl] = pbbfu_[ikl] + zincflx[iw*ncol+iplon]*zfu[iw*(nlayers+1)*ncol+jk*ncol+iplon];				
                    pbbfd_[ikl] = pbbfd_[ikl] + zincflx[iw*ncol+iplon]*zfd[iw*(nlayers+1)*ncol+jk*ncol+iplon];
                    pbbcu_[ikl] = pbbcu_[ikl] + zincflx[iw*ncol+iplon]*zcu[iw*(nlayers+1)*ncol+jk*ncol+iplon];
                    pbbcd_[ikl] = pbbcd_[ikl] + zincflx[iw*ncol+iplon]*zcd[iw*(nlayers+1)*ncol+jk*ncol+iplon];

					if (idelm == 0)
					{
						pbbfddir_[ikl] = pbbfddir_[ikl]  + zincflx[iw*ncol+iplon]*ztdbt_nodel[jk*ncol+iplon];
						pbbcddir_[ikl] = pbbcddir_[ikl] + zincflx[iw*ncol+iplon]*ztdbtc_nodel[jk*ncol+iplon];
					}
					else if (idelm == 1)
					{
						pbbfddir_[ikl]  = pbbfddir_[ikl]  + zincflx[iw*ncol+iplon]*ztdbt[jk*ncol+iplon];
						pbbcddir_[ikl] = pbbcddir_[ikl] + zincflx[iw*ncol+iplon]*ztdbtc[jk*ncol+iplon];
					}

					if (ibm >= 9 && ibm <= 12)
					{
						puvcd[ikl*ncol+iplon] = puvcd[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*zcd[iw*(nlayers+1)*ncol+jk*ncol+iplon];
						puvfd[ikl*ncol+iplon] = puvfd[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*zfd[iw*(nlayers+1)*ncol+jk*ncol+iplon];

						if (idelm == 0)
						{
							puvfddir[ikl*ncol+iplon] = puvfddir[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*ztdbt_nodel[jk*ncol+iplon];
							puvcddir[ikl*ncol+iplon] = puvcddir[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*ztdbtc_nodel[jk*ncol+iplon];
						}
						else if (idelm == 1)
						{
							puvfddir[ikl*ncol+iplon] = puvfddir[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*ztdbt[jk*ncol+iplon];
							puvcddir[ikl*ncol+iplon] = puvcddir[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*ztdbtc[jk*ncol+iplon];
						}
					}
					else if (ibm == 8)
					{
						puvcd[ikl*ncol+iplon] = puvcd[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*zcd[iw*(nlayers+1)*ncol+jk*ncol+iplon];
						puvfd[ikl*ncol+iplon] = puvfd[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*zfd[iw*(nlayers+1)*ncol+jk*ncol+iplon];
						pnicd[ikl*ncol+iplon] = pnicd[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*zcd[iw*(nlayers+1)*ncol+jk*ncol+iplon];
						pnifd[ikl*ncol+iplon] = pnifd[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*zfd[iw*(nlayers+1)*ncol+jk*ncol+iplon];

						if (idelm ==  0)
						{
							puvfddir[ikl*ncol+iplon] = puvfddir[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*ztdbt_nodel[jk*ncol+iplon];
							puvcddir[ikl*ncol+iplon] = puvcddir[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*ztdbtc_nodel[jk*ncol+iplon];
							pnifddir[ikl*ncol+iplon] = pnifddir[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*ztdbt_nodel[jk*ncol+iplon];
							pnicddir[ikl*ncol+iplon] = pnicddir[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*ztdbtc_nodel[jk*ncol+iplon];
						}
						else if (idelm == 1)
						{
							puvfddir[ikl*ncol+iplon] = puvfddir[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*ztdbt[jk*ncol+iplon];
							puvcddir[ikl*ncol+iplon] = puvcddir[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*ztdbtc[jk*ncol+iplon];
							pnifddir[ikl*ncol+iplon] = pnifddir[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*ztdbt[jk*ncol+iplon];
							pnicddir[ikl*ncol+iplon] = pnicddir[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*ztdbtc[jk*ncol+iplon];
						}

						pnicu[ikl*ncol+iplon] = pnicu[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*zcu[iw*(nlayers+1)*ncol+jk*ncol+iplon];
						pnifu[ikl*ncol+iplon] = pnifu[ikl*ncol+iplon] + 0.5*zincflx[iw*ncol+iplon]*zfu[iw*(nlayers+1)*ncol+jk*ncol+iplon];
					}
					else if(ibm == 13 || ibm <= 7)
					{
						pnicd[ikl*ncol+iplon] = pnicd[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*zcd[iw*(nlayers+1)*ncol+jk*ncol+iplon];
						pnifd[ikl*ncol+iplon] = pnifd[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*zfd[iw*(nlayers+1)*ncol+jk*ncol+iplon];

						if (idelm == 0)
						{
							pnifddir[ikl*ncol+iplon] = pnifddir[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*ztdbt_nodel[jk*ncol+iplon];
							pnicddir[ikl*ncol+iplon] = pnicddir[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*ztdbtc_nodel[jk*ncol+iplon];
						}
						else if (idelm == 1)
						{
						 pnifddir[ikl*ncol+iplon] = pnifddir[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*ztdbt[jk*ncol+iplon];
						 pnicddir[ikl*ncol+iplon] = pnicddir[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*ztdbtc[jk*ncol+iplon];
						}

						pnicu[ikl*ncol+iplon] = pnicu[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*zcu[iw*(nlayers+1)*ncol+jk*ncol+iplon];
						pnifu[ikl*ncol+iplon] = pnifu[ikl*ncol+iplon] + zincflx[iw*ncol+iplon]*zfu[iw*(nlayers+1)*ncol+jk*ncol+iplon];
					}
				}
			}
        }
		for(jk=0;jk<klev+1;jk++)
		{
			pbbfu[jk*ncol+iplon]=pbbfu_[jk];
			pbbfd[jk*ncol+iplon]=pbbfd_[jk];
			pbbcu[jk*ncol+iplon]=pbbcu_[jk];
			pbbcd[jk*ncol+iplon]=pbbcd_[jk];
			pbbcddir[jk*ncol+iplon]=pbbcddir_[jk];
			pbbfddir[jk*ncol+iplon]=pbbfddir_[jk];			
		}
	}
}

//主函数
int main(void)
{
	int icld,inflgsw,iceflgsw,liqflgsw;
	const int mxlay=203;
	const int mg=16;
	const int naerec=6;
	int klev=52;
	int g,iplon,ig,k,i,j,iw,jb,jg,jl,ibm,igt;
	double zepzen;
	int icpr,istart,iend,iaer,idelm,ims;
	double oneminus,avogad,grav;//pi,
	int dyofyr=3;
	double adjes=1.e-6;
	int  ib1,ib2;
	int lchnk;
    int iout;
    double bpade;

	clock_t starts;
	clock_t ends;



//----- Input -----
	double *play;
	cudaMallocHost((double**) &play,nlay*ncol*sizeof(double));
	double *plev;
	cudaMallocHost((double**) &plev,nlayers*ncol*sizeof(double));
	double *tlay;
	cudaMallocHost((double**) &tlay,nlay*ncol*sizeof(double));
	double *tlev;
	cudaMallocHost((double**) &tlev,nlayers*ncol*sizeof(double));
	double *tsfc;
	cudaMallocHost((double**) &tsfc,ncol*sizeof(double));
	double *h2ovmr;
	cudaMallocHost((double**) &h2ovmr,nlay*ncol*sizeof(double));
	double *o3vmr;
	cudaMallocHost((double**) &o3vmr,nlay*ncol*sizeof(double));
	double *co2vmr;
	cudaMallocHost((double**) &co2vmr,nlay*ncol*sizeof(double));
	double *ch4vmr;
	cudaMallocHost((double**) &ch4vmr,nlay*ncol*sizeof(double));
	double *o2vmr;
	cudaMallocHost((double**) &o2vmr,nlay*ncol*sizeof(double));
	double *n2ovmr;
	cudaMallocHost((double**) &n2ovmr,nlay*ncol*sizeof(double));
	double *solvar;
	cudaMallocHost((double**) &solvar,jpb2*ncol*sizeof(double));
	double *cldfmcl;
	cudaMallocHost((double**) &cldfmcl,nlay*ncol*ngptsw*sizeof(double));
	double *taucmcl;
	cudaMallocHost((double**) &taucmcl,nlay*ncol*ngptsw*sizeof(double));
	double *ssacmcl;
	cudaMallocHost((double**) &ssacmcl,nlay*ncol*ngptsw*sizeof(double));
	double *asmcmcl;
	cudaMallocHost((double**) &asmcmcl,nlay*ncol*ngptsw*sizeof(double));
	double *fsfcmcl;
	cudaMallocHost((double**) &fsfcmcl,nlay*ncol*ngptsw*sizeof(double));
	double *ciwpmcl;
	cudaMallocHost((double**) &ciwpmcl,nlay*ncol*ngptsw*sizeof(double));
	double *clwpmcl;
	cudaMallocHost((double**) &clwpmcl,nlay*ncol*ngptsw*sizeof(double));
	double *reicmcl;
	cudaMallocHost((double**) &reicmcl,nlay*ncol*sizeof(double));
	double *relqmcl;
	cudaMallocHost((double**) &relqmcl,nlay*ncol*sizeof(double));
	double *tauaer;
	cudaMallocHost((double**) &tauaer,nlay*ncol*nbndsw*sizeof(double));
	double *ssaaer;
	cudaMallocHost((double**) &ssaaer,nlay*ncol*nbndsw*sizeof(double));
	double *asmaer;
	cudaMallocHost((double**) &asmaer,nlay*ncol*nbndsw*sizeof(double));


	//inatm out
	double *pavel_c;
	cudaMallocHost((double**) &pavel_c,nlayers*ncol*sizeof(double));
	double *tavel_c;
	cudaMallocHost((double**) &tavel_c,nlayers*ncol*sizeof(double));
	double *pz_c;
	cudaMallocHost((double**) &pz_c,(nlayers+1)*ncol*sizeof(double));
	double *tz_c;
	cudaMallocHost((double**) &tz_c,(nlayers+1)*ncol*sizeof(double));
	double *tbound_c;
	cudaMallocHost((double**) &tbound_c,ncol*sizeof(double));
	double *pdp_c;
	cudaMallocHost((double**) &pdp_c,nlayers*ncol*sizeof(double));
	double *coldry_c;
	cudaMallocHost((double**) &coldry_c,nlayers*ncol*sizeof(double));
	double *wkl_c;
	cudaMallocHost((double**) &wkl_c,nlayers*mxmol*ncol*sizeof(double));
	double *adjflux_c;
	cudaMallocHost((double**) &adjflux_c,jpband*ncol*sizeof(double));
	double *taua_c;
	cudaMallocHost((double**) &taua_c,nlayers*ncol*nbndsw*sizeof(double));
	double *ssaa_c;
	cudaMallocHost((double**) &ssaa_c,nlayers*ncol*nbndsw*sizeof(double));
	double *asma_c;
	cudaMallocHost((double**) &asma_c,nlayers*ncol*nbndsw*sizeof(double));

	double *cldfmc_c;
	cudaMallocHost((double**) &cldfmc_c,nlayers*ncol*ngptsw*sizeof(double));
	double *taucmc_c;
	cudaMallocHost((double**) &taucmc_c,nlayers*ncol*ngptsw*sizeof(double));
	double *ssacmc_c;
	cudaMallocHost((double**) &ssacmc_c,nlayers*ncol*ngptsw*sizeof(double));
	double *asmcmc_c;
	cudaMallocHost((double**) &asmcmc_c,nlayers*ncol*ngptsw*sizeof(double));
	double *fsfcmc_c;
	cudaMallocHost((double**) &fsfcmc_c,nlayers*ncol*ngptsw*sizeof(double));
	double *ciwpmc_c;
	cudaMallocHost((double**) &ciwpmc_c,nlayers*ncol*ngptsw*sizeof(double));
	double *clwpmc_c;
	cudaMallocHost((double**) &clwpmc_c,nlayers*ncol*ngptsw*sizeof(double));
	double *reicmc_c;
	cudaMallocHost((double**) &reicmc_c,nlayers*ncol*sizeof(double));
	double *dgesmc_c;
	cudaMallocHost((double**) &dgesmc_c,nlayers*ncol*sizeof(double));
	double *relqmc_c;
	cudaMallocHost((double**) &relqmc_c,nlayers*ncol*sizeof(double));


//------------------cldprmc out
	double *taormc_c;
	cudaMallocHost((double**) &taormc_c,nlayers*ncol*ngptsw*sizeof(double));


//------------------cldprmc_in-------
		double *extliq1_c;
	cudaMallocHost((double**) &extliq1_c,14*58*sizeof(double));
	double *ssaliq1_c;
	cudaMallocHost((double**) &ssaliq1_c,14*58*sizeof(double));
	double *asyliq1_c;
	cudaMallocHost((double**) &asyliq1_c,14*58*sizeof(double));
	double *extice2_c;
	cudaMallocHost((double**) &extice2_c,14*43*sizeof(double));
	double *ssaice2_c;
	cudaMallocHost((double**) &ssaice2_c,14*43*sizeof(double));
	double *asyice2_c;
	cudaMallocHost((double**) &asyice2_c,14*43*sizeof(double));
	double *extice3_c;
	cudaMallocHost((double**) &extice3_c,14*46*sizeof(double));
	double *ssaice3_c;
	cudaMallocHost((double**) &ssaice3_c,14*46*sizeof(double));
	double *asyice3_c;
	cudaMallocHost((double**) &asyice3_c,14*46*sizeof(double));
	double *fdlice3_c;
	cudaMallocHost((double**) &fdlice3_c,14*46*sizeof(double));
	double *abari_c;
	cudaMallocHost((double**) &abari_c,5*sizeof(double));
	double *bbari_c;
	cudaMallocHost((double**) &bbari_c,5*sizeof(double));
	double *cbari_c;
	cudaMallocHost((double**) &cbari_c,5*sizeof(double));
	double *dbari_c;
	cudaMallocHost((double**) &dbari_c,5*sizeof(double));
	double *ebari_c;
	cudaMallocHost((double**) &ebari_c,5*sizeof(double));
	double *fbari_c;
	cudaMallocHost((double**) &fbari_c,5*sizeof(double));
	double *wavenum2_c;
	cudaMallocHost((double**) &wavenum2_c,14*sizeof(double));
	double *ngb_c;
	cudaMallocHost((double**) &ngb_c,112*sizeof(double));
//--------------setcoef out---
	int *laytrop_c;
	cudaMallocHost((int**) &laytrop_c,ncol*sizeof(int));
	int *layswtch_c;
	cudaMallocHost((int**) &layswtch_c,ncol*sizeof(int));
	int *laylow_c;
	cudaMallocHost((int**) &laylow_c,ncol*sizeof(int));
	int *jp_c;
	cudaMallocHost((int**) &jp_c,nlayers*ncol*sizeof(int));
	int *jt_c;
	cudaMallocHost((int**) &jt_c,nlayers*ncol*sizeof(int));
	int *jt1_c;
	cudaMallocHost((int**) &jt1_c,nlayers*ncol*sizeof(int));
	int *indself_c;
	cudaMallocHost((int**) &indself_c,nlayers*ncol*sizeof(int));
	int *indfor_c;
	cudaMallocHost((int**) &indfor_c,nlayers*ncol*sizeof(int));
	double *colmol_c;
	cudaMallocHost((double**) &colmol_c,nlayers*ncol*sizeof(double));
	double *co2mult_c;
	cudaMallocHost((double**) &co2mult_c,nlayers*ncol*sizeof(double));
	double *colh2o_c;
	cudaMallocHost((double**) &colh2o_c,nlayers*ncol*sizeof(double));
	double *colco2_c;
	cudaMallocHost((double**) &colco2_c,nlayers*ncol*sizeof(double));
	double *colo3_c;
	cudaMallocHost((double**) &colo3_c,nlayers*ncol*sizeof(double));
	double *coln2o_c;
	cudaMallocHost((double**) &coln2o_c,nlayers*ncol*sizeof(double));
	double *colch4_c;
	cudaMallocHost((double**) &colch4_c,nlayers*ncol*sizeof(double));
	double *colo2_c;
	cudaMallocHost((double**) &colo2_c,nlayers*ncol*sizeof(double));
	double *selffac_c;
	cudaMallocHost((double**) &selffac_c,nlayers*ncol*sizeof(double));
	double *selffrac_c;
	cudaMallocHost((double**) &selffrac_c,nlayers*ncol*sizeof(double));
	double *forfac_c;
	cudaMallocHost((double**) &forfac_c,nlayers*ncol*sizeof(double));
	double *forfrac_c;
	cudaMallocHost((double**) &forfrac_c,nlayers*ncol*sizeof(double));
	double *fac00_c;
	cudaMallocHost((double**) &fac00_c,nlayers*ncol*sizeof(double));
	double *fac01_c;
	cudaMallocHost((double**) &fac01_c,nlayers*ncol*sizeof(double));
	double *fac10_c;
	cudaMallocHost((double**) &fac10_c,nlayers*ncol*sizeof(double));
	double *fac11_c;
	cudaMallocHost((double**) &fac11_c,nlayers*ncol*sizeof(double));


//----------------------setcoef in
	double *preflog_c;
	cudaMallocHost((double**) &preflog_c,59*sizeof(double));
	double *tref_c;
	cudaMallocHost((double**) &tref_c,59*sizeof(double));

//--------------------spcvmc in
	double *albdif;
	cudaMallocHost((double**) &albdif,nbndsw*ncol*sizeof(double));
    double *albdir;
	cudaMallocHost((double**) &albdir,nbndsw*ncol*sizeof(double));
    double *cossza;
	cudaMallocHost((double**) &cossza,ncol*sizeof(double));
    double *zcldfmc;
	cudaMallocHost((double**) &zcldfmc,ngptsw*(nlay+1)*sizeof(double));
    double *ztaucmc;
	cudaMallocHost((double**) &ztaucmc,ngptsw*(nlay+1)*sizeof(double));
    double *zasycmc;
	cudaMallocHost((double**) &zasycmc,ngptsw*(nlay+1)*sizeof(double));
    double *zomgcmc;
	cudaMallocHost((double**) &zomgcmc,ngptsw*(nlay+1)*sizeof(double));
    double *ztaormc;
	cudaMallocHost((double**) &ztaormc,ngptsw*(nlay+1)*sizeof(double));
    double *ztaua;
	cudaMallocHost((double**) &ztaua,nbndsw*(nlay+1)*ncol*sizeof(double));
    double *zasya;
	cudaMallocHost((double**) &zasya,nbndsw*(nlay+1)*ncol*sizeof(double));
    double *zomga;
	cudaMallocHost((double**) &zomga,nbndsw*(nlay+1)*ncol*sizeof(double));

	int *ngs_c;
	cudaMallocHost((int**) &ngs_c,14*sizeof(int));
    int *ngc_c;
	cudaMallocHost((int**) &ngc_c,14*sizeof(int));
	double *exp_tbl_c;
	cudaMallocHost((double**) &exp_tbl_c,10001*sizeof(double));

//---------------------taumol use
	double *absa16_c;
	cudaMallocHost((double**) &absa16_c,6*585*sizeof(double));
	double *absb16_c;
	cudaMallocHost((double**) &absb16_c,6*235*sizeof(double));
	double *selfref16_c;
	cudaMallocHost((double**) &selfref16_c,6*10*sizeof(double));
	double *forref16_c;
	cudaMallocHost((double**) &forref16_c,6*3*sizeof(double));
	double *sfluxref16_c;
	cudaMallocHost((double**) &sfluxref16_c,6*sizeof(double));

	double *absa17_c;
	cudaMallocHost((double**) &absa17_c,12*585*sizeof(double));
	double *absb17_c;
	cudaMallocHost((double**) &absb17_c,12*1175*sizeof(double));
	double *selfref17_c;
	cudaMallocHost((double**) &selfref17_c,12*10*sizeof(double));
	double *forref17_c;
	cudaMallocHost((double**) &forref17_c,12*4*sizeof(double));
	double *sfluxref17_c;
	cudaMallocHost((double**) &sfluxref17_c,5*12*sizeof(double));

	double *absa18_c;
	cudaMallocHost((double**) &absa18_c,8*585*sizeof(double));
	double *absb18_c;
	cudaMallocHost((double**) &absb18_c,8*235*sizeof(double));
	double *selfref18_c;
	cudaMallocHost((double**) &selfref18_c,8*10*sizeof(double));
	double *forref18_c;
	cudaMallocHost((double**) &forref18_c,8*3*sizeof(double));
	double *sfluxref18_c;
	cudaMallocHost((double**) &sfluxref18_c,9*8*sizeof(double));

	double *absa19_c;
	cudaMallocHost((double**) &absa19_c,8*585*sizeof(double));
	double *absb19_c;
	cudaMallocHost((double**) &absb19_c,8*235*sizeof(double));
	double *selfref19_c;
	cudaMallocHost((double**) &selfref19_c,8*10*sizeof(double));
	double *forref19_c;
	cudaMallocHost((double**) &forref19_c,8*3*sizeof(double));
	double *sfluxref19_c;
	cudaMallocHost((double**) &sfluxref19_c,9*8*sizeof(double));

	double *absa20_c;
	cudaMallocHost((double**) &absa20_c,10*65*sizeof(double));
	double *absb20_c;
	cudaMallocHost((double**) &absb20_c,10*235*sizeof(double));
	double *selfref20_c;
	cudaMallocHost((double**) &selfref20_c,10*10*sizeof(double));
	double *forref20_c;
	cudaMallocHost((double**) &forref20_c,10*4*sizeof(double));
	double *sfluxref20_c;
	cudaMallocHost((double**) &sfluxref20_c,10*sizeof(double));
	double *absch420_c;
	cudaMallocHost((double**) &absch420_c,10*sizeof(double));

	double *absa21_c;
	cudaMallocHost((double**) &absa21_c,10*585*sizeof(double));
	double *absb21_c;
	cudaMallocHost((double**) &absb21_c,10*1175*sizeof(double));
	double *selfref21_c;
	cudaMallocHost((double**) &selfref21_c,10*10*sizeof(double));
	double *forref21_c;
	cudaMallocHost((double**) &forref21_c,10*4*sizeof(double));
	double *sfluxref21_c;
	cudaMallocHost((double**) &sfluxref21_c,9*10*sizeof(double));

	double *absa22_c;
	cudaMallocHost((double**) &absa22_c,2*585*sizeof(double));
	double *absb22_c;
	cudaMallocHost((double**) &absb22_c,2*235*sizeof(double));
	double *selfref22_c;
	cudaMallocHost((double**) &selfref22_c,2*10*sizeof(double));
	double *forref22_c;
	cudaMallocHost((double**) &forref22_c,2*3*sizeof(double));
	double *sfluxref22_c;
	cudaMallocHost((double**) &sfluxref22_c,9*2*sizeof(double));

	double *absa23_c;
	cudaMallocHost((double**) &absa23_c,10*65*sizeof(double));
	double *selfref23_c;
	cudaMallocHost((double**) &selfref23_c,10*10*sizeof(double));
	double *forref23_c;
	cudaMallocHost((double**) &forref23_c,10*3*sizeof(double));
	double *sfluxref23_c;
	cudaMallocHost((double**) &sfluxref23_c,10*sizeof(double));
	double *rayl23_c;
	cudaMallocHost((double**) &rayl23_c,10*sizeof(double));

	double *absa24_c;
	cudaMallocHost((double**) &absa24_c,8*585*sizeof(double));
	double *absb24_c;
	cudaMallocHost((double**) &absb24_c,8*235*sizeof(double));
	double *selfref24_c;
	cudaMallocHost((double**) &selfref24_c,8*10*sizeof(double));
	double *forref24_c;
	cudaMallocHost((double**) &forref24_c,8*3*sizeof(double));
	double *sfluxref24_c;
	cudaMallocHost((double**) &sfluxref24_c,9*8*sizeof(double));
	double *abso3a24_c;
	cudaMallocHost((double**) &abso3a24_c,8*sizeof(double));
	double *abso3b24_c;
	cudaMallocHost((double**) &abso3b24_c,8*sizeof(double));
	double *rayla24_c;
	cudaMallocHost((double**) &rayla24_c,9*8*sizeof(double));
	double *raylb24_c;
	cudaMallocHost((double**) &raylb24_c,8*sizeof(double));

	double *absa25_c;
	cudaMallocHost((double**) &absa25_c,6*65*sizeof(double));
	double *sfluxref25_c;
	cudaMallocHost((double**) &sfluxref25_c,6*sizeof(double));
	double *abso3a25_c;
	cudaMallocHost((double**) &abso3a25_c,6*sizeof(double));
	double *abso3b25_c;
	cudaMallocHost((double**) &abso3b25_c,6*sizeof(double));
	double *rayl25_c;
	cudaMallocHost((double**) &rayl25_c,6*sizeof(double));

	double *sfluxref26_c;
	cudaMallocHost((double**) &sfluxref26_c,6*sizeof(double));
	double *rayl26_c;
	cudaMallocHost((double**) &rayl26_c,6*sizeof(double));

	double *absa27_c;
	cudaMallocHost((double**) &absa27_c,8*65*sizeof(double));
	double *absb27_c;
	cudaMallocHost((double**) &absb27_c,8*235*sizeof(double));
	double *sfluxref27_c;
	cudaMallocHost((double**) &sfluxref27_c,8*sizeof(double));
	double *rayl27_c;
	cudaMallocHost((double**) &rayl27_c,8*sizeof(double));

	double *absa28_c;
	cudaMallocHost((double**) &absa28_c,6*585*sizeof(double));
	double *absb28_c;
	cudaMallocHost((double**) &absb28_c,6*1175*sizeof(double));
	double *sfluxref28_c;
	cudaMallocHost((double**) &sfluxref28_c,5*6*sizeof(double));

	double *absa29_c;
	cudaMallocHost((double**) &absa29_c,12*65*sizeof(double));
	double *absb29_c;
	cudaMallocHost((double**) &absb29_c,12*235*sizeof(double));
	double *selfref29_c;
	cudaMallocHost((double**) &selfref29_c,12*10*sizeof(double));
	double *forref29_c;
	cudaMallocHost((double**) &forref29_c,12*4*sizeof(double));
	double *sfluxref29_c;
	cudaMallocHost((double**) &sfluxref29_c,12*sizeof(double));
	double *absh2o29_c;
	cudaMallocHost((double**) &absh2o29_c,12*sizeof(double));
	double *absco229_c;
	cudaMallocHost((double**) &absco229_c,12*sizeof(double));


//taumol out------------------------
	double *zsflxzen_c;          // solar source function
    cudaMallocHost((double**) &zsflxzen_c, ngptsw*ncol*sizeof(double));                        ////  Dimensions: (ngptsw)
    double *ztaug_c;            // gaseous optical depth
    cudaMallocHost((double**) &ztaug_c,ngptsw*nlayers*ncol*sizeof(double));                        //   Dimensions: (nlayers,ngptsw)
    double *ztaur_c;
	cudaMallocHost((double**) &ztaur_c,ngptsw*nlayers*ncol*sizeof(double));
//taumol use---------------------
	int *lrtchkclr_c;
	cudaMallocHost((int**) &lrtchkclr_c,nlayers*ncol*sizeof(int));
	int *lrtchkcld_c;
	cudaMallocHost((int**) &lrtchkcld_c,nlayers*ncol*sizeof(int));

	double *ztauc_c;
	cudaMallocHost((double**) &ztauc_c,nlayers*ncol*sizeof(double));
	double *zomcc_c;
	cudaMallocHost((double**) &zomcc_c,nlayers*ncol*sizeof(double));
	double *zgcc_c;
	cudaMallocHost((double**) &zgcc_c,nlayers*ncol*sizeof(double));
//----------------------spcvmc out
	double *zbbcd;
	cudaMallocHost((double**) &zbbcd,(nlay+2)*ncol*sizeof(double));
    double *zbbcu;
	cudaMallocHost((double**) &zbbcu,(nlay+2)*ncol*sizeof(double));
    double *zbbfd;
	cudaMallocHost((double**) &zbbfd,(nlay+2)*ncol*sizeof(double));
    double *zbbfu;
	cudaMallocHost((double**) &zbbfu,(nlay+2)*ncol*sizeof(double));
    double *zbbfddir;
	cudaMallocHost((double**) &zbbfddir,(nlay+2)*ncol*sizeof(double));
    double *zbbcddir;
	cudaMallocHost((double**) &zbbcddir,(nlay+2)*ncol*sizeof(double));
    double *zuvcd;
	cudaMallocHost((double**) &zuvcd,(nlay+2)*ncol*sizeof(double));
    double *zuvfd;
	cudaMallocHost((double**) &zuvfd,(nlay+2)*ncol*sizeof(double));
    double *zuvcddir;
	cudaMallocHost((double**) &zuvcddir,(nlay+2)*ncol*sizeof(double));
    double *zuvfddir;
	cudaMallocHost((double**) &zuvfddir,(nlay+2)*ncol*sizeof(double));
    double *znicd;
	cudaMallocHost((double**) &znicd,(nlay+2)*ncol*sizeof(double));
    double *znifd;
	cudaMallocHost((double**) &znifd,(nlay+2)*ncol*sizeof(double));
    double *znicddir;
	cudaMallocHost((double**) &znicddir,(nlay+2)*ncol*sizeof(double));
    double *znifddir;
	cudaMallocHost((double**) &znifddir,(nlay+2)*ncol*sizeof(double));
    double *znicu;
	cudaMallocHost((double**) &znicu,(nlay+2)*ncol*sizeof(double));
	double *znifu;
	cudaMallocHost((double**) &znifu,(nlay+2)*ncol*sizeof(double));

	int *nspa_c;     //nspa_c={9,9,9,9,1,9,9,1,9,1,0,1,9,1};
	cudaMallocHost((int**) &nspa_c,14*sizeof(int));
	int *nspb_c;  		//nspb_c={1,5,1,1,1,5,1,0,1,0,0,1,5,1};
	cudaMallocHost((int**) &nspb_c,14*sizeof(int));

	//定义设备端使用的指针数组
	//inatm_in
	double *play_d;
	double *plev_d;
	double *tlay_d;
	double *tlev_d;
	double *tsfc_d;
	double *h2ovmr_d;
	double *o3vmr_d;
    double *co2vmr_d;
    double *ch4vmr_d;
	double *o2vmr_d;
	double *n2ovmr_d;
	double *solvar_d;
	double *cldfmcl_d;
	double *taucmcl_d;
	double *ssacmcl_d;
	double *asmcmcl_d;
	double *fsfcmcl_d;
	double *ciwpmcl_d;
	double *clwpmcl_d;
	double *reicmcl_d;
	double *relqmcl_d;
	double *tauaer_d;
	double *ssaaer_d;
	double *asmaer_d;

	//inatm_d-out
	double *pavel;
	double *pz;
	double *pdp;
	double *tavel;
	double *tz;
	double *tbound;
	double *adjflux;
	double *wkl;
	double *coldry;
	double *cldfmc;
	double *taucmc;
	double *ssacmc;
	double *asmcmc;
	double *fsfcmc;
	double *ciwpmc;
	double *clwpmc;
	double *reicmc;
	double *dgesmc;
	double *relqmc;
	double *taua;
	double *ssaa;
	double *asma;

	//cldprmc-out
	double *taormc;

	//cldprmc-use
	double *extliq1;
	double *ssaliq1;
	double *asyliq1;
	double *extice2;
	double *ssaice2;
	double *asyice2;
	double *extice3;
	double *ssaice3;
	double *asyice3;
	double *fdlice3;
	double *abari;
	double *bbari;
	double *cbari;
	double *dbari;
	double *ebari;
	double *fbari;
	double *wavenum2;  //16-29
	double *ngb;

	//setcoef out

	int *laytrop;
	int *layswtch;
	int *laylow;
	int *jp;
	int *jt;
	int *jt1;
	int *indself;
	int *indfor;
	double *colmol;
	double *co2mult;
	double *colh2o;
	double *colco2;
	double *colo3;
	double *coln2o;
	double *colch4;
	double *colo2;
	double *selffac;
	double *selffrac;
	double *forfac;
	double *forfrac;
	double *fac00;
	double *fac01;
	double *fac10;
	double *fac11;

	// setcoef use
	double *preflog;
	double *tref;

	//spcvmc_in    ***********

	double *palbd_d;
	double *palbp_d;
	double *prmu0_d;
	double *pcldfmc_d;
	double *ptaucmc_d;
	double *pasycmc_d;
	double *pomgcmc_d;
	double *ptaormc_d;
	double *ptaua_d;
	double *pasya_d;
	double *pomga_d;

	//spcvmc——out------------------------
	double *pbbcd;
	double *pbbcu;
	double *pbbfd;
	double *pbbfu;
	double *pbbfddir;
	double *pbbcddir;
	double *puvcd;
	double *puvfd;
	double *puvcddir;
	double *puvfddir;
	double *pnicd;
	double *pnifd;
	double *pnicddir;
	double *pnifddir;
	double *pnicu;
	double *pnifu;

	int *ngc;
	int *ngs;
	double *exp_tbl;

	//spcvmc_vrtqdr_out
	double *zcd;
	double *zcu;
	double *zfd;
	double *zfu;

	///vrtqdr-in
	double *zrefc;
	double *zrefdc;
	double *ztrac;
	double *ztradc;

	//reftra in
	double *zgcc;
	double *ztauc;
	double *zomcc;
	int *lrtchkclr,*lrtchkcld;

	//taumolout
	double *ztaug;
	double *ztaur;
	double *zsflxzen;

	//taumol use
	int *nspa;
	int *nspb;

	double *absa16;
	double *absb16;
	double *selfref16;
	double *forref16;
	double *sfluxref16;

	double *absa17;
	double *absb17;
	double *selfref17;
	double *forref17;
	double *sfluxref17;

	double *absa18;
	double *absb18;
	double *selfref18;
	double *forref18;
	double *sfluxref18;

	double *absa19;
	double *absb19;
	double *selfref19;
	double *forref19;
	double *sfluxref19;

	double *absa20;
	double *absb20;
	double *selfref20;
	double *forref20;
	double *sfluxref20;
	double *absch420;

	double *absa21;
	double *absb21;
	double *selfref21;
	double *forref21;
	double *sfluxref21;

	double *absa22;
	double *absb22;
	double *selfref22;
	double *forref22;
	double *sfluxref22;

	double *absa23;
	double *selfref23;
	double *forref23;
	double *sfluxref23;
	double *rayl23;

	double *absa24;
	double *absb24;
	double *selfref24;
	double *forref24;
	double *sfluxref24;
	double *abso3a24;
	double *abso3b24;
	double *rayla24;
	double *raylb24;

	double *absa25;
	double *sfluxref25;
	double *abso3a25;
	double *abso3b25;
	double *rayl25;

	double *sfluxref26;
	double *rayl26;

	double *absa27;
	double *absb27;
	double *sfluxref27;
	double *rayl27;

	double *absa28;
	double *absb28;
	double *sfluxref28;

	double *absa29;
	double *absb29;
	double *selfref29;
	double *forref29;
	double *sfluxref29;
	double *absh2o29;
	double *absco229;

	//原来为local，统一了数组名
	double *zomco,*ztrao,*zrefdo,*zgco;
	double *zref,*zrupd,*zdbt,*zrefo;
	double *zrup,*ztrado,*zrdnd,*ztdbt;
	double *ztauo,*zrupc,*zrupdc,*zrdndc;
	double *ztdbtc,*zdbtc,*zrefd,*ztra,*ztrad;
	double *zincflx,*ztdbtc_nodel,*ztdbt_nodel;
	double *zdbtc_nodel;

	cudaMalloc((void**)&play_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&plev_d,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&tlay_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&tlev_d,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&tsfc_d,ncol*sizeof(double));
	cudaMalloc((void**)&h2ovmr_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&o3vmr_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&co2vmr_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&ch4vmr_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&o2vmr_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&n2ovmr_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&solvar_d,jpb2*ncol*sizeof(double));
	cudaMalloc((void**)&cldfmcl_d,nlay*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&taucmcl_d,nlay*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&ssacmcl_d,nlay*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&asmcmcl_d,nlay*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&fsfcmcl_d,nlay*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&ciwpmcl_d,nlay*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&clwpmcl_d,nlay*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&reicmcl_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&relqmcl_d,nlay*ncol*sizeof(double));
	cudaMalloc((void**)&tauaer_d,nbndsw*nlay*ncol*sizeof(double));
	cudaMalloc((void**)&ssaaer_d,nbndsw*nlay*ncol*sizeof(double));
	cudaMalloc((void**)&asmaer_d,nbndsw*nlay*ncol*sizeof(double));

	cudaMalloc((void**)&pavel,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&pz,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pdp,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&tavel,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&tz,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&tbound,ncol*sizeof(double));
	cudaMalloc((void**)&adjflux,jpband*ncol*sizeof(double));
	cudaMalloc((void**)&wkl,nlayers*mxmol*ncol*sizeof(double));
	cudaMalloc((void**)&coldry,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&cldfmc,nlayers*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&taucmc,nlayers*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&ssacmc,nlayers*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&asmcmc,nlayers*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&fsfcmc,nlayers*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&ciwpmc,nlayers*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&clwpmc,nlayers*ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&reicmc,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&dgesmc,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&relqmc,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&taua,nbndsw*nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&ssaa,nbndsw*nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&asma,nbndsw*nlayers*ncol*sizeof(double));

	cudaMalloc((void**)&taormc,nlayers*ngptsw*ncol*sizeof(double));

	cudaMalloc((void**)&extliq1,14*58*sizeof(double));
	cudaMalloc((void**)&ssaliq1,14*58*sizeof(double));
	cudaMalloc((void**)&asyliq1,14*58*sizeof(double));
	cudaMalloc((void**)&extice2,14*43*sizeof(double));
	cudaMalloc((void**)&ssaice2,14*43*sizeof(double));
	cudaMalloc((void**)&asyice2,14*43*sizeof(double));
	cudaMalloc((void**)&extice3,14*46*sizeof(double));
	cudaMalloc((void**)&ssaice3,14*46*sizeof(double));
	cudaMalloc((void**)&asyice3,14*46*sizeof(double));
	cudaMalloc((void**)&fdlice3,14*46*sizeof(double));
	cudaMalloc((void**)&abari,5*sizeof(double));
	cudaMalloc((void**)&bbari,5*sizeof(double));
	cudaMalloc((void**)&cbari,5*sizeof(double));
	cudaMalloc((void**)&dbari,5*sizeof(double));
	cudaMalloc((void**)&ebari,5*sizeof(double));
	cudaMalloc((void**)&fbari,5*sizeof(double));
	cudaMalloc((void**)&wavenum2,14*sizeof(double));
	cudaMalloc((void**)&ngb,112*sizeof(double));

	cudaMalloc((void**)&laytrop,ncol*sizeof(int));
	cudaMalloc((void**)&layswtch,ncol*sizeof(int));
	cudaMalloc((void**)&laylow,ncol*sizeof(int));
	cudaMalloc((void**)&jp,nlayers*ncol*sizeof(int));
	cudaMalloc((void**)&jt,nlayers*ncol*sizeof(int));
	cudaMalloc((void**)&jt1,nlayers*ncol*sizeof(int));
	cudaMalloc((void**)&indself,nlayers*ncol*sizeof(int));
	cudaMalloc((void**)&indfor,nlayers*ncol*sizeof(int));
	cudaMalloc((void**)&colmol,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&co2mult,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&colh2o,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&colco2,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&colo3,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&coln2o,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&colch4,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&colo2,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&selffac,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&selffrac,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&forfac,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&forfrac,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&fac00,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&fac01,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&fac10,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&fac11,nlayers*ncol*sizeof(double));

	cudaMalloc((void**)&preflog,59*sizeof(double));
	cudaMalloc((void**)&tref,59*sizeof(double));

	cudaMalloc((void**)&palbd_d,nbndsw*ncol*sizeof(double));
	cudaMalloc((void**)&palbp_d,nbndsw*ncol*sizeof(double));
	cudaMalloc((void**)&prmu0_d,ncol*sizeof(double));
	cudaMalloc((void**)&pcldfmc_d,ngptsw*nlayers*sizeof(double));
	cudaMalloc((void**)&ptaucmc_d,ngptsw*nlayers*sizeof(double));
	cudaMalloc((void**)&pasycmc_d,ngptsw*nlayers*sizeof(double));
	cudaMalloc((void**)&pomgcmc_d,ngptsw*nlayers*sizeof(double));
	cudaMalloc((void**)&ptaormc_d,ngptsw*nlayers*sizeof(double));
	cudaMalloc((void**)&ptaua_d,nbndsw*nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&pasya_d,nbndsw*nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&pomga_d,nbndsw*nlayers*ncol*sizeof(double));

	cudaMalloc((void**)&pbbcd,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pbbcu,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pbbfd,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pbbfu,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pbbfddir,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pbbcddir,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&puvcd,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&puvfd,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&puvcddir,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&puvfddir,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pnicd,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pnifd,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pnicddir,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pnifddir,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pnicu,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&pnifu,(nlayers+1)*ncol*sizeof(double));

	cudaMalloc((void**)&ngc,14*sizeof(int));
	cudaMalloc((void**)&ngs,14*sizeof(int));
	cudaMalloc((void**)&exp_tbl,10001*sizeof(double));

	cudaMalloc((void**)&zcd,ngptsw*(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zcu,ngptsw*(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zfd,ngptsw*(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zfu,ngptsw*(nlayers+1)*ncol*sizeof(double));

	cudaMalloc((void**)&zrefc,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrefdc,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&ztrac,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&ztradc,(nlayers+1)*ncol*sizeof(double));

	cudaMalloc((void**)&zgcc,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&ztauc,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&zomcc,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&lrtchkclr,nlayers*ncol*sizeof(int));
	cudaMalloc((void**)&lrtchkcld,nlayers*ncol*sizeof(int));

	cudaMalloc((void**)&ztaug,ngptsw*nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&ztaur,ngptsw*nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&zsflxzen,ngptsw*ncol*sizeof(double));

	cudaMalloc((void**)&nspa,14*sizeof(int));
	cudaMalloc((void**)&nspb,14*sizeof(int));

	cudaMalloc((void**)&absa16,6*585*sizeof(double));
	cudaMalloc((void**)&absb16,6*235*sizeof(double));
	cudaMalloc((void**)&selfref16,6*10*sizeof(double));
	cudaMalloc((void**)&forref16,6*3*sizeof(double));
	cudaMalloc((void**)&sfluxref16,6*sizeof(double));

	cudaMalloc((void**)&absa17,12*585*sizeof(double));
	cudaMalloc((void**)&absb17,12*1175*sizeof(double));
	cudaMalloc((void**)&selfref17,12*10*sizeof(double));
	cudaMalloc((void**)&forref17,12*4*sizeof(double));
	cudaMalloc((void**)&sfluxref17,5*12*sizeof(double));

	cudaMalloc((void**)&absa18,8*585*sizeof(double));
	cudaMalloc((void**)&absb18,8*235*sizeof(double));
	cudaMalloc((void**)&selfref18,8*10*sizeof(double));
	cudaMalloc((void**)&forref18,8*3*sizeof(double));
	cudaMalloc((void**)&sfluxref18,9*8*sizeof(double));

	cudaMalloc((void**)&absa19,8*585*sizeof(double));
	cudaMalloc((void**)&absb19,8*235*sizeof(double));
	cudaMalloc((void**)&selfref19,8*10*sizeof(double));
	cudaMalloc((void**)&forref19,8*3*sizeof(double));
	cudaMalloc((void**)&sfluxref19,9*8*sizeof(double));

	cudaMalloc((void**)&absa20,10*65*sizeof(double));
	cudaMalloc((void**)&absb20,10*235*sizeof(double));
	cudaMalloc((void**)&selfref20,10*10*sizeof(double));
	cudaMalloc((void**)&forref20,10*4*sizeof(double));
	cudaMalloc((void**)&sfluxref20,10*sizeof(double));
	cudaMalloc((void**)&absch420,10*sizeof(double));

	cudaMalloc((void**)&absa21,10*585*sizeof(double));
	cudaMalloc((void**)&absb21,10*1175*sizeof(double));
	cudaMalloc((void**)&selfref21,10*10*sizeof(double));
	cudaMalloc((void**)&forref21,10*4*sizeof(double));
	cudaMalloc((void**)&sfluxref21,9*10*sizeof(double));

	cudaMalloc((void**)&absa22,2*585*sizeof(double));
	cudaMalloc((void**)&absb22,2*235*sizeof(double));
	cudaMalloc((void**)&selfref22,2*10*sizeof(double));
	cudaMalloc((void**)&forref22,2*3*sizeof(double));
	cudaMalloc((void**)&sfluxref22,9*2*sizeof(double));

	cudaMalloc((void**)&absa23,10*65*sizeof(double));
	cudaMalloc((void**)&selfref23,10*10*sizeof(double));
	cudaMalloc((void**)&forref23,10*3*sizeof(double));
	cudaMalloc((void**)&sfluxref23,10*sizeof(double));
	cudaMalloc((void**)&rayl23,10*sizeof(double));

	cudaMalloc((void**)&absa24,8*585*sizeof(double));
	cudaMalloc((void**)&absb24,8*235*sizeof(double));
	cudaMalloc((void**)&selfref24,8*10*sizeof(double));
	cudaMalloc((void**)&forref24,8*3*sizeof(double));
	cudaMalloc((void**)&sfluxref24,9*8*sizeof(double));
	cudaMalloc((void**)&abso3a24,8*sizeof(double));
	cudaMalloc((void**)&abso3b24,8*sizeof(double));
	cudaMalloc((void**)&rayla24,9*8*sizeof(double));
	cudaMalloc((void**)&raylb24,8*sizeof(double));

	cudaMalloc((void**)&absa25,6*65*sizeof(double));
	cudaMalloc((void**)&sfluxref25,6*sizeof(double));
	cudaMalloc((void**)&abso3a25,6*sizeof(double));
	cudaMalloc((void**)&abso3b25,6*sizeof(double));
	cudaMalloc((void**)&rayl25,6*sizeof(double));

	cudaMalloc((void**)&sfluxref26,6*sizeof(double));
	cudaMalloc((void**)&rayl26,6*sizeof(double));

	cudaMalloc((void**)&absa27,8*65*sizeof(double));
	cudaMalloc((void**)&absb27,8*235*sizeof(double));
	cudaMalloc((void**)&sfluxref27,8*sizeof(double));
	cudaMalloc((void**)&rayl27,8*sizeof(double));

	cudaMalloc((void**)&absa28,6*585*sizeof(double));
	cudaMalloc((void**)&absb28,6*1175*sizeof(double));
	cudaMalloc((void**)&sfluxref28,5*6*sizeof(double));

	cudaMalloc((void**)&absa29,12*65*sizeof(double));
	cudaMalloc((void**)&absb29,12*235*sizeof(double));
	cudaMalloc((void**)&selfref29,12*10*sizeof(double));
	cudaMalloc((void**)&forref29,12*4*sizeof(double));
	cudaMalloc((void**)&sfluxref29,12*sizeof(double));
	cudaMalloc((void**)&absh2o29,12*sizeof(double));
	cudaMalloc((void**)&absco229,12*sizeof(double));

	cudaMalloc((void**)&zomco,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&ztrao,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrefdo,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zgco,nlayers*ncol*sizeof(double));
	cudaMalloc((void**)&zref,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrupd,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zdbt,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrefo,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrup,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&ztrado,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrdnd,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&ztdbt,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&ztauo,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrupc,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrupdc,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrdndc,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&ztdbtc,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zdbtc,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zrefd,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&ztra,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&ztrad,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zincflx,ngptsw*ncol*sizeof(double));
	cudaMalloc((void**)&ztdbtc_nodel,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&ztdbt_nodel,(nlayers+1)*ncol*sizeof(double));
	cudaMalloc((void**)&zdbtc_nodel,(nlayers+1)*ncol*sizeof(double));



	starts=clock();

	inflgsw=0;
        iceflgsw=0;
        liqflgsw=0;
        icld=2;

	//ngs_c={6,18,26,34,44,54,56,66,74,80,86,94,100,112};
	ngs_c[0]=6;
	ngs_c[1]=18;
	ngs_c[2]=26;
	ngs_c[3]=34;
	ngs_c[4]=44;
	ngs_c[5]=54;
	ngs_c[6]=56;
	ngs_c[7]=66;
	ngs_c[8]=74;
	ngs_c[9]=80;
	ngs_c[10]=86;
	ngs_c[11]=94;
	ngs_c[12]=100;
	ngs_c[13]=112;

	 //ngc_c={6,12,8,8,10,10,2,10,8,6,6,8,6,12};
	ngc_c[0]=6;
	ngc_c[1]=12;
	ngc_c[2]=8;
	ngc_c[3]=8;
	ngc_c[4]=10;
	ngc_c[5]=10;
	ngc_c[6]=2;
	ngc_c[7]=10;
	ngc_c[8]=8;
	ngc_c[9]=6;
	ngc_c[10]=6;
	ngc_c[11]=8;
	ngc_c[12]=6;
	ngc_c[13]=12;

	//nspa_c={9,9,9,9,1,9,9,1,9,1,0,1,9,1};
	nspa_c[0]=9;
	nspa_c[1]=9;
	nspa_c[2]=9;
	nspa_c[3]=9;
	nspa_c[4]=1;
	nspa_c[5]=9;
	nspa_c[6]=9;
	nspa_c[7]=1;
	nspa_c[8]=9;
	nspa_c[9]=1;
	nspa_c[10]=0;
	nspa_c[11]=1;
	nspa_c[12]=9;
	nspa_c[13]=1;

	//nspb_c={1,5,1,1,1,5,1,0,1,0,0,1,5,1};
	nspb_c[0]=1;
	nspb_c[1]=5;
	nspb_c[2]=1;
	nspb_c[3]=1;
	nspb_c[4]=1;
	nspb_c[5]=5;
	nspb_c[6]=1;
	nspb_c[7]=0;
	nspb_c[8]=1;
	nspb_c[9]=0;
	nspb_c[10]=0;
	nspb_c[11]=1;
	nspb_c[12]=5;
	nspb_c[13]=1;
//!!!!inaslize!!!!!!()()(()(()()(()(()()()()()()()()))))

	for(i=0;i<ncol;i++)
		cossza[i]=1;

	for(i=0;i<nlay;i++)
	{
		for(j=0;j<ncol;j++)
		{
			play[i*ncol+j]=(j+1)+10;
			tlay[i*ncol+j]=(j+1)+10;
			h2ovmr[i*ncol+j]=(j+1)+10;
			o3vmr[i*ncol+j]=(j+1)+10;
			co2vmr[i*ncol+j]=(j+1)+10;
			ch4vmr[i*ncol+j]=(j+1)+10;
			o2vmr[i*ncol+j]=(j+1)+10;
			n2ovmr[i*ncol+j]=(j+1)+10;
			reicmcl[i*ncol+j]=(j+1)+10;
			relqmcl[i*ncol+j]=(j+1)+10;
		}
	}
	for(i=0;i<nlay+1;i++)
	{
		for(j=0;j<ncol;j++)
		{
			plev[i*ncol+j]=(i+1)+(j+1);
			tlev[i*ncol+j]=(j+1)+10;
		}
	}
	for(i=0;i<ncol;i++)
	{
		tsfc[i]=(i+1)+10;
	}
	for(i=0;i<nlay;i++)
	{
		for(j=0;j<ngptsw;j++)
		{
			for(k=0;k<ncol;k++)
			{
				cldfmcl[i*ngptsw*ncol+j*ncol+k]=(k+1)+(j+1);
				ciwpmcl[i*ngptsw*ncol+j*ncol+k]=(k+1)+(j+1);
				clwpmcl[i*ngptsw*ncol+j*ncol+k]=(k+1)+(j+1);
				taucmcl[i*ngptsw*ncol+j*ncol+k]=(k+1)+(j+1);
				ssacmcl[i*ngptsw*ncol+j*ncol+k]=(k+1)+(j+1);
				asmcmcl[i*ngptsw*ncol+j*ncol+k]=(k+1)+(j+1);
				fsfcmcl[i*ngptsw*ncol+j*ncol+k]=(k+1)+(j+1);

			}
		}
	}
	for(i=0;i<jpb2;i++)
	{
		for(j=0;j<ncol;j++)
		{
			solvar[i*ncol+j]=(i+1)+(j+1);
		}
	}
	for(i=0;i<nbndsw;i++)
	{
		for(j=0;j<nlay;j++)
		{
			for(k=0;k<ncol;k++)
			{
				tauaer[i*nlay*ncol+j*ncol+k]=(k+1)+(j+1);
				ssaaer[i*nlay*ncol+j*ncol+k]=(k+1)+(j+1);
				asmaer[i*nlay*ncol+j*ncol+k]=(k+1)+(j+1);
			}
		}
	}
	for(i=0;i<14;i++)
	{
		for(j=0;j<58;j++)
		{
			extliq1_c[i*58+j]=(j+1)+10;
			ssaliq1_c[i*58+j]=(j+1)+10;
			asyliq1_c[i*58+j]=(j+1)+10;

		}
		for(j=0;j<43;j++)
		{
			extice2_c[i*43+j]=(j+1)+10;
			ssaice2_c[i*43+j]=(j+1)+10;
			asyice2_c[i*43+j]=(j+1)+10;

		}
		for(j=0;j<46;j++)
		{
			extice3_c[i*46+j]=(j+1)+10;
			ssaice3_c[i*46+j]=(j+1)+10;
			asyice3_c[i*46+j]=(j+1)+10;
			fdlice3_c[i*46+j]=(j+1)+10;
		}

	}
	for(i=0;i<5;i++)
	{
		abari_c[i]=i+1;
		bbari_c[i]=i+1;
		cbari_c[i]=(i+1)+10;
		dbari_c[i]=i+1;
		ebari_c[i]=i+1;
		fbari_c[i]=(i+1)+10;
	}
	for(i=0;i<14;i++)
	{
		wavenum2_c[i]=(i+16)+10;
	}
	for(i=0;i<112;i++)
	{
		ngb_c[i]=i+1;
	}

	for(i=0;i<585;i++)
	{
		for(j=0;j<6;j++)
		{
			absa16_c[j*585+i]=(j+1);
		}
	}
	for(i=0;i<235;i++)
	{
		for(j=0;j<6;j++)
		{
			absb16_c[j*235+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<6;j++)
		{
			selfref16_c[j*10+i]=(j+1);
		}
	}
	for(i=0;i<3;i++)
	{
		for(j=0;j<6;j++)
		{
			forref16_c[j*3+i]=(j+1);
		}
	}
	for(i=0;i<6;i++)
	{
		sfluxref16_c[i]=i+1;
	}
	for(i=0;i<585;i++)
	{
		for(j=0;j<12;j++)
		{
			absa17_c[j*585+i]=(j+1);
		}
	}
	for(i=0;i<1175;i++)
	{
		for(j=0;j<12;j++)
		{
			absb17_c[j*1175+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<12;j++)
		{
			selfref17_c[j*10+i]=(j+1);
		}
	}
	for(i=0;i<4;i++)
	{
		for(j=0;j<12;j++)
		{
			forref17_c[j*4+i]=(j+1);
		}
	}
	for(i=0;i<12;i++)
	{
		for(j=0;j<5;j++)
		{
			sfluxref17_c[j*12+i]=(i+1);
		}
	}
	for(i=0;i<585;i++)
	{
		for(j=0;j<8;j++)
		{
			absa18_c[j*585+i]=(j+1);
			absa19_c[j*585+i]=(j+1);
		}
	}
	for(i=0;i<235;i++)
	{
		for(j=0;j<8;j++)
		{
			absb18_c[j*235+i]=(j+1);
			absb19_c[j*235+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<8;j++)
		{
			selfref18_c[j*10+i]=(j+1);
			selfref19_c[j*10+i]=(j+1);
		}
	}
	for(i=0;i<3;i++)
	{
		for(j=0;j<8;j++)
		{
			forref18_c[j*3+i]=(j+1);
			forref19_c[j*3+i]=(j+1);
		}
	}
	for(i=0;i<8;i++)
	{
		for(j=0;j<9;j++)
		{
			sfluxref18_c[j*8+i]=(i+1);
			sfluxref19_c[j*8+i]=(i+1);
		}
	}
	for(i=0;i<65;i++)
	{
		for(j=0;j<10;j++)
		{
			absa20_c[j*65+i]=(j+1);
		}
	}
	for(i=0;i<235;i++)
	{
		for(j=0;j<10;j++)
		{
			absb20_c[j*235+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<10;j++)
		{
			selfref20_c[j*10+i]=(j+1);
		}
	}
	for(i=0;i<4;i++)
	{
		for(j=0;j<10;j++)
		{
			forref20_c[j*4+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		sfluxref20_c[i]=(i+1);
		absch420_c[i]=(i+1);
	}
	for(i=0;i<585;i++)
	{
		for(j=0;j<10;j++)
		{
			absa21_c[j*585+i]=(j+1);
		}
	}
	for(i=0;i<1175;i++)
	{
		for(j=0;j<10;j++)
		{
			absb21_c[j*1175+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<10;j++)
		{
			selfref21_c[j*10+i]=(j+1);
		}
	}
	for(i=0;i<4;i++)
	{
		for(j=0;j<10;j++)
		{
			forref21_c[j*4+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<9;j++)
		{
			sfluxref21_c[j*10+i]=(i+1);
		}
	}
	for(i=0;i<585;i++)
	{
		for(j=0;j<2;j++)
		{
			absa22_c[j*585+i]=(j+1);
		}
	}
	for(i=0;i<235;i++)
	{
		for(j=0;j<2;j++)
		{
			absb22_c[j*235+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<2;j++)
		{
			selfref22_c[j*10+i]=(j+1);
		}
	}
	for(i=0;i<3;i++)
	{
		for(j=0;j<2;j++)
		{
			forref22_c[j*3+i]=(j+1);
		}
	}
	for(i=0;i<2;i++)
	{
		for(j=0;j<9;j++)
		{
			sfluxref22_c[j*2+i]=(i+1);
		}
	}
	for(i=0;i<65;i++)
	{
		for(j=0;j<10;j++)
		{
			absa23_c[j*65+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<10;j++)
		{
			selfref23_c[j*10+i]=(j+1);
		}
	}
	for(i=0;i<3;i++)
	{
		for(j=0;j<10;j++)
		{
			forref23_c[j*3+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		sfluxref23_c[i]=i+1;
		rayl23_c[i]=i+1;
	}
	for(i=0;i<585;i++)
	{
		for(j=0;j<8;j++)
		{
			absa24_c[j*585+i]=(j+1);
		}
	}
	for(i=0;i<235;i++)
	{
		for(j=0;j<8;j++)
		{
			absb24_c[j*235+i]=(j+1);
		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<8;j++)
		{
			selfref24_c[j*10+i]=(j+1);
		}
	}
	for(i=0;i<3;i++)
	{
		for(j=0;j<8;j++)
		{
			forref24_c[j*3+i]=(j+1);
		}
	}
	for(i=0;i<8;i++)
	{
		for(j=0;j<9;j++)
		{
			sfluxref24_c[j*8+i]=(i+1);
		}
	}
	for(i=0;i<8;i++)
	{
		abso3a24_c[i]=(i+1);
		abso3b24_c[i]=(i+1);
		raylb24_c[i]=(i+1);
	}
	for(i=0;i<8;i++)
	{
		for(j=0;j<9;j++)
		{
			rayla24_c[j*8+i]=(j+1);
		}
	}
	for(i=0;i<65;i++)
	{
		for(j=0;j<6;j++)
		{
			absa25_c[j*65+i]=(j+1);
		}
	}
	for(i=0;i<6;i++)
	{
		sfluxref25_c[i]=(i+1);
		abso3a25_c[i]=(i+1);
		abso3b25_c[i]=(i+1);
		rayl25_c[i]=(i+1);
		sfluxref26_c[i]=(i+1);
		rayl26_c[i]=(i+1);
	}
	for(i=0;i<65;i++)
	{
		for(j=0;j<8;j++)
		{
			absa27_c[j*65+i]=(j+1);
		}
	}
	for(i=0;i<235;i++)
	{
		for(j=0;j<8;j++)
		{
			absb27_c[j*235+i]=(j+1);
		}
	}
	for(i=0;i<8;i++)
	{
		sfluxref27_c[i]=(i+1);
		rayl27_c[i]=(i+1);
	}
	for(i=0;i<585;i++)
	{
		for(j=0;j<6;j++)
		{
			absa28_c[j*585+i]=(j+1);
		}
	}
	for(i=0;i<1175;i++)
	{
		for(j=0;j<6;j++)
		{
			absb28_c[j*1175+i]=(j+1);
		}
	}
	for(i=0;i<6;i++)
	{
		for(j=0;j<5;j++)
		{
			sfluxref28_c[j*6+i]=(i+1);
		}
	}
	for(i=0;i<65;i++)
	{
		for(j=0;j<12;j++)
		{
			absa29_c[j*65+i]=(j+1);

		}
	}
	for(i=0;i<235;i++)
	{
		for(j=0;j<12;j++)
		{
			absb29_c[j*235+i]=(j+1);

		}
	}
	for(i=0;i<10;i++)
	{
		for(j=0;j<12;j++)
		{
			 selfref29_c[j*10+i]=(j+1);
		}
	}
	for(i=0;i<4;i++)
	{
		for(j=0;j<12;j++)
		{
			forref29_c[j*4+i]=(j+1);
		}
	}
	for(i=0;i<12;i++)
	{
		sfluxref29_c[i]=(i+1);
		absh2o29_c[i]=(i+1);
		absco229_c[i]=(i+1);
	}


	for(i=0;i<59;i++)
	{
		preflog_c[i]=i+1;
		tref_c[i]=i+1;
	}

	for(i=0;i<ncol;i++)
	{

		for(j=0;j<nbndsw;j++)
		{
			albdif[j*ncol+i]=5;
			albdir[j*ncol+i]=5;
		}
	}
	for(j=0;j<nlayers;j++)
		{
			for(k=0;k<ngptsw;k++)
			{
				zcldfmc[k*nlayers+j]=1;
				ztaucmc[k*nlayers+j]=1;
				zasycmc[k*nlayers+j]=1;
				zomgcmc[k*nlayers+j]=1;
				ztaormc[k*nlayers+j]=1;
			}
		}
	for(i=0;i<ncol;i++)
	{
		for(j=0;j<nlayers;j++)
		{
			for(k=0;k<nbndsw;k++)
			{
				ztaua[k*nlayers*ncol+j*ncol+i]=5;
				zasya[k*nlayers*ncol+j*ncol+i]=5;
				zomga[k*nlayers*ncol+j*ncol+i]=5;
			}
		}
	}

	for(i=0;i<=10000;i++)
	{
		exp_tbl_c[i]=0.5;
	}

//Initializationswrite
//-----------------------------------------------------------------
	dim3 grid1,tBlock1,grid2,tBlock2,grid3,tBlock3,grid4,tBlock4,grid5,tBlock5;

	//二维并行
	tBlock1.x=128; //112
	tBlock1.y=4;
	tBlock1.z=1;
	grid1.x=ceil((ncol*1.0)/tBlock1.x);
	grid1.y=ceil((nlayers*1.0)/tBlock1.y);
	grid1.z=1;

	//三维并行
	tBlock2.x=2;
	tBlock2.y=128;
	tBlock2.z=2;
	grid2.x=ceil((ngptsw*1.0)/tBlock2.x);
	grid2.y=ceil((ncol*1.0)/tBlock2.y);
	grid2.z=ceil((nlayers)/tBlock2.z);

	tBlock3.x=2;
	tBlock3.y=128;
	tBlock3.z=2;
	grid3.x=ceil((ngptsw*1.0)/tBlock3.x);
	grid3.y=ceil((ncol*1.0)/tBlock3.y);
	grid3.z=ceil((nlayers*1.0+2)/tBlock3.z);


//-----------------------------------------------------------------

		if(icld<0 || icld>3)
			icld=2;
		iaer=10;

		idelm=1;
		lchnk=0;
		zepzen = 1.e-10;
		//zepsec = 1.e-06_r8
        //zepzen = 1.e-10_r8
		oneminus=1.0-1.e-6;
	//	pi=2.0*asin(1.0);
		istart=16;
		iend=29;
		icpr=0;
		ims=2;
		iout = 0;
		//-------set by zc
		avogad=2.e4;
		grav=9.8;
		bpade=1.0;
	g=0;
	float elapsedTime1,elapsedTime2,elapsedTime3,elapsedTime4,elapsedTime5,inatmtime=0.0,cldprmctime=0.0,setcoeftime=0.0,taumoltime=0.0,spcvmctime=0.0;
	float chuanshutime=0.0,chuanshutime1=0.0,chuanshutime2=0.0,chuanshutime3=0.0;

	
	while(g<384)
	{
		//inatm
		cudaEvent_t nstart11,nstop11;
		cudaEventCreate(&nstart11);
		cudaEventCreate(&nstop11);
		cudaEventRecord(nstart11);
		cudaMemcpy(play_d, play, nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(plev_d,plev,nlayers*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(tlay_d, tlay, nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(tlev_d, tlev, nlayers*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(tsfc_d, tsfc, ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(h2ovmr_d, h2ovmr, nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(o3vmr_d, o3vmr, nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(co2vmr_d, co2vmr, nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ch4vmr_d, ch4vmr, nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(o2vmr_d, o2vmr, nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(n2ovmr_d, n2ovmr, nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(solvar_d, solvar,jpb2*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(cldfmcl_d,cldfmcl,nlay*ncol*ngptsw*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(taucmcl_d,taucmcl,nlay*ncol*ngptsw*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ssacmcl_d,ssacmcl,nlay*ncol*ngptsw*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(asmcmcl_d,asmcmcl,nlay*ncol*ngptsw*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(fsfcmcl_d,fsfcmcl,nlay*ncol*ngptsw*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ciwpmcl_d,ciwpmcl,nlay*ncol*ngptsw*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(clwpmcl_d,clwpmcl,nlay*ncol*ngptsw*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(reicmcl_d,reicmcl,nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(relqmcl_d,relqmcl,nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(tauaer_d,tauaer,nbndsw*nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ssaaer_d,ssaaer,nbndsw*nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(asmaer_d,asmaer,nbndsw*nlay*ncol*sizeof(double),cudaMemcpyHostToDevice);
		cudaEventRecord(nstop11);
		
		cudaEventSynchronize(nstop11);

		cudaEventElapsedTime(&chuanshutime1, nstart11, nstop11);
		chuanshutime+=chuanshutime1;

		//cldprmc
		cudaEvent_t nstart22,nstop22;
		cudaEventCreate(&nstart22);
		cudaEventCreate(&nstop22);
		cudaEventRecord(nstart22);
		cudaMemcpy(extliq1, extliq1_c,14*58*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ssaliq1, ssaliq1_c,14*58*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(asyliq1, asyliq1_c,14*58*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(extice2, extice2_c,14*43*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ssaice2, ssaice2_c,14*43*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(asyice2, asyice2_c,14*43*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(extice3, extice3_c,14*46*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ssaice3, ssaice3_c,14*46*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(asyice3, asyice3_c,14*46*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(fdlice3, fdlice3_c,14*46*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(abari, abari_c,5*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(bbari, bbari_c,5*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(cbari, cbari_c,5*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(dbari, dbari_c,5*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ebari, ebari_c,5*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(fbari, fbari_c,5*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(wavenum2, wavenum2_c,14*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(ngb, ngb_c,112*sizeof(double),cudaMemcpyHostToDevice);

		
		//setcoef
		cudaMemcpy(preflog, preflog_c,59*sizeof(double),cudaMemcpyHostToDevice);
		cudaMemcpy(tref, tref_c,59*sizeof(double),cudaMemcpyHostToDevice);

		cudaEventRecord(nstop22);
		
		cudaEventSynchronize(nstop22);

		cudaEventElapsedTime(&chuanshutime1, nstart22, nstop22);
		chuanshutime+=chuanshutime1;

		//taumol
		cudaEvent_t nstart33,nstop33;
		cudaEventCreate(&nstart33);
		cudaEventCreate(&nstop33);
		cudaEventRecord(nstart33);
			cudaMemcpy(nspa,nspa_c,14*sizeof(int),cudaMemcpyHostToDevice);
			cudaMemcpy(nspb,nspb_c,14*sizeof(int),cudaMemcpyHostToDevice);
			cudaMemcpy(absa16,absa16_c,6*585*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb16,absb16_c,6*235*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref16,selfref16_c,6*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref16,forref16_c,6*3*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref16,sfluxref16_c,6*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa17,absa17_c,12*585*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb17,absb17_c,12*1175*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref17,selfref17_c,12*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref17,forref17_c,12*4*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref17,sfluxref17_c,5*12*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa18,absa18_c,8*585*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb18,absb18_c,8*235*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref18,selfref18_c,8*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref18,forref18_c,8*3*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref18,sfluxref18_c,9*8*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa19,absa19_c,8*585*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb19,absb19_c,8*235*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref19,selfref19_c,8*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref19,forref19_c,8*3*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref19,sfluxref19_c,9*8*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa20,absa20_c,10*65*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb20,absb20_c,10*235*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref20,selfref20_c,10*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref20,forref20_c,10*4*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref20,sfluxref20_c,10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absch420,absch420_c,10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa21,absa21_c,10*585*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb21,absb21_c,10*1175*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref21,selfref21_c,10*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref21,forref21_c,10*4*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref21,sfluxref21_c,9*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa22,absa22_c,2*585*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb22,absb22_c,2*235*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref22,selfref22_c,2*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref22,forref22_c,2*3*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref22,sfluxref22_c,9*2*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa23,absa23_c,10*65*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref23,selfref23_c,10*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref23,forref23_c,10*3*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref23,sfluxref23_c,10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(rayl23,rayl23_c,10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa24,absa24_c,8*585*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb24,absb24_c,8*235*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref24,selfref24_c,8*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref24,forref24_c,8*3*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref24,sfluxref24_c,9*8*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(abso3a24,abso3a24_c,8*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(abso3b24,abso3b24_c,8*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(rayla24,rayla24_c,9*8*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(raylb24,raylb24_c,8*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa25,absa25_c,6*65*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref25,sfluxref25_c,6*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(abso3a25,abso3a25_c,6*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(abso3b25,abso3b25_c,6*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(rayl25,rayl25_c,6*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref26,sfluxref26_c,6*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(rayl26,rayl26_c,6*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa27,absa27_c,8*65*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb27,absb27_c,8*235*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref27,sfluxref27_c,8*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(rayl27,rayl27_c,8*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa28,absa28_c,6*585*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb28,absb28_c,6*1175*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref28,sfluxref28_c,5*6*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absa29,absa29_c,12*65*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absb29,absb29_c,12*235*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(selfref29,selfref29_c,12*10*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(forref29,forref29_c,12*4*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(sfluxref29,sfluxref29_c,12*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absh2o29,absh2o29_c,12*sizeof(double),cudaMemcpyHostToDevice);
			cudaMemcpy(absco229,absco229_c,12*sizeof(double),cudaMemcpyHostToDevice);
	
			cudaEventRecord(nstop33);
			
			cudaEventSynchronize(nstop33);
	
			cudaEventElapsedTime(&chuanshutime1, nstart33, nstop33);
			chuanshutime+=chuanshutime1;


		cudaEvent_t nstart1,nstop1;
		cudaEventCreate(&nstart1);
		cudaEventCreate(&nstop1);
		cudaEventRecord(nstart1);	
        inatm_d1<<<grid1,tBlock1>>>(wkl,
				    reicmc,dgesmc,
				    relqmc,taua,ssaa,asma);
						
        inatm_d2<<<grid1,tBlock1>>>(play_d,plev_d,tlay_d,tlev_d,
				    h2ovmr_d,o3vmr_d,co2vmr_d,ch4vmr_d,
				    o2vmr_d,n2ovmr_d,
				    pavel,pz,pdp,
				    tavel,tz,wkl);					
        inatm_d3<<<grid2,tBlock2>>>(icld,cldfmcl_d,taucmcl_d,
				    ssacmcl_d,asmcmcl_d,fsfcmcl_d,ciwpmcl_d,
				    clwpmcl_d,reicmcl_d,relqmcl_d,tauaer_d,
				    cldfmc,taucmc,ssacmc,asmcmc,
				    fsfcmc,ciwpmc,clwpmc);
				
		 inatm_d4<<<grid1,tBlock1>>>(icld,iaer,inflgsw,iceflgsw,liqflgsw,
					     reicmcl_d,relqmcl_d,tauaer_d,
					     ssaaer_d,asmaer_d,
					     reicmc,dgesmc,
					     relqmc,taua,ssaa,asma);

        inatm_d5<<<ceil(ncol*1.0/(tpb*4)),(tpb*4)>>>(avogad,grav,
						     adjes,dyofyr,
						     tsfc_d,
						     solvar_d,
						     pavel,pz,pdp,
						     tavel,tz,tbound,adjflux,wkl,
						     coldry,cldfmc,taucmc,ssacmc,asmcmc,
						     fsfcmc,ciwpmc,clwpmc,reicmc,dgesmc,
						     relqmc,taua,ssaa,asma);
					
		cudaEventRecord(nstop1);
		
		cudaEventSynchronize(nstop1);

		cudaEventElapsedTime(&elapsedTime1, nstart1, nstop1);
		inatmtime+=elapsedTime1;

		cudaEvent_t nstart2,nstop2;
		cudaEventCreate(&nstart2);
		cudaEventCreate(&nstop2);
		cudaEventRecord(nstart2);	
		cldprmc_d<<<grid3,tBlock3>>>(taormc,taucmc,ciwpmc,
					     clwpmc,cldfmc,fsfcmc,ssacmc,asmcmc,
					     reicmc,wavenum2,abari,bbari,
					     cbari,dbari,ebari,fbari,
					     ngb,extice2,ssaice2,asyice2,
					     dgesmc,extice3,ssaice3,asyice3,
					     fdlice3,relqmc,extliq1,ssaliq1,
					     asyliq1);
					
		cudaEventRecord(nstop2);
		
		cudaEventSynchronize(nstop2);

		cudaEventElapsedTime(&elapsedTime2, nstart2, nstop2);
		cldprmctime+=elapsedTime2;
		
		cudaEvent_t nstart3,nstop3;
		cudaEventCreate(&nstart3);
		cudaEventCreate(&nstop3);
		cudaEventRecord(nstart3);
		setcoef_sw1<<<grid1,tBlock1>>>(tbound,tz,laytrop,
						layswtch,laylow,pavel,jp,preflog,jt,tavel,tref,
						jt1,wkl,coldry,forfac,indfor,forfrac,colh2o,colco2,
						colo3,coln2o,colch4,colo2,colmol,co2mult,selffac,
						selffrac,indself,fac10,fac00,fac11,fac01);

		setcoef_sw2<<<ceil(ncol*1.0/(tpb*4)),(tpb*4)>>>(laytrop,
														layswtch,laylow,pavel);

													
		cudaEventRecord(nstop3);
		
		cudaEventSynchronize(nstop3);

		cudaEventElapsedTime(&elapsedTime3, nstart3, nstop3);
		setcoeftime+=elapsedTime3;

		cudaEvent_t nstart4,nstop4;
		cudaEventCreate(&nstart4);
		cudaEventCreate(&nstop4);
		cudaEventRecord(nstart4);

		taumol_sw<<<grid1,tBlock1>>>(oneminus, laytrop,colh2o,colch4,fac00,fac10,fac01,fac11,
			jp,jt,nspa,jt1,indself,indfor,colmol,ztaug,absa16,selffac,
		   selfref16,selffrac,forfac,forref16,forfrac,ztaur,nspb,absb16,zsflxzen,
		   sfluxref16,colco2,absa17,selfref17,forref17,absb17,sfluxref17,absa18,
		   selfref18,forref18,sfluxref18,absb18,absa19,selfref19,forref19,sfluxref19,
		   absb19,absa20,selfref20,forref20,absch420,sfluxref20,absb20,absa21,
		   selfref21,forref21,sfluxref21,absb21,colo2,absa22,selfref22,forref22,
		   sfluxref22,absb22,rayl23,absa23,selfref23,forref23,sfluxref23,rayla24,
		   absa24,colo3,abso3a24,selfref24,forref24,sfluxref24,absb24,abso3b24,
		   rayl25,absa25,abso3a25,sfluxref25,sfluxref26,rayl26,rayl27,absa27,
		   absb27,sfluxref27,absa28,absb28,sfluxref28,absa29,selfref29,forref29,
		   absco229,absb29,absh2o29,sfluxref29,abso3b25,raylb24);


		cudaEventRecord(nstop4);
		
		cudaEventSynchronize(nstop4);

		cudaEventElapsedTime(&elapsedTime4, nstart4, nstop4);
		taumoltime+=elapsedTime4;


		cudaStream_t stream5[4];
		for( i=0;i<4;++i)
		cudaStreamCreate(&stream5[i]);

		int offset=0;
		for( i=0;i<4;++i){
			cudaEvent_t nstart44,nstop44;
			cudaEventCreate(&nstart44);
			cudaEventCreate(&nstop44);
			cudaEventRecord(nstart44);
		
			cudaMemcpyAsync(&prmu0_d[i*(ncol/4)],&cossza[i*(ncol/4)],ncol*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&palbd_d[i*(nbndsw*ncol/4)],&albdif[i*(nbndsw*ncol/4)],nbndsw*ncol*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&palbp_d[i*(nbndsw*ncol/4)],&albdir[i*(nbndsw*ncol/4)],nbndsw*ncol*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&pcldfmc_d[i*(ngptsw*nlayers/4)],&zcldfmc[i*(ngptsw*nlayers/4)],ngptsw*nlayers*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&ptaucmc_d[i*(ngptsw*nlayers/4)],&ztaucmc[i*(ngptsw*nlayers/4)],ngptsw*nlayers*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&pasycmc_d[i*(ngptsw*nlayers/4)],&zasycmc[i*(ngptsw*nlayers/4)],ngptsw*nlayers*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&pomgcmc_d[i*(ngptsw*nlayers/4)],&zomgcmc[i*(ngptsw*nlayers/4)],ngptsw*nlayers*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&ptaormc_d[i*(ngptsw*nlayers/4)],&ztaormc[i*(ngptsw*nlayers/4)],ngptsw*nlayers*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&ptaua_d[i*(nbndsw*nlayers*ncol/4)],&ztaua[i*(nbndsw*nlayers*ncol/4)],nbndsw*nlayers*ncol*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&pasya_d[i*(nbndsw*nlayers*ncol/4)],&zasya[i*(nbndsw*nlayers*ncol/4)],nbndsw*nlayers*ncol*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(&pomga_d[i*(nbndsw*nlayers*ncol/4)],&zomga[i*(nbndsw*nlayers*ncol/4)],nbndsw*nlayers*ncol*sizeof(double)/4,cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(exp_tbl,exp_tbl_c,10001*sizeof(double),cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(ngs,ngs_c,14*sizeof(int),cudaMemcpyHostToDevice,stream5[i]);
			cudaMemcpyAsync(ngc,ngc_c,14*sizeof(int),cudaMemcpyHostToDevice,stream5[i]);

		cudaEventRecord(nstop44);
		
		cudaEventSynchronize(nstop44);

		cudaEventElapsedTime(&chuanshutime1, nstart44, nstop44);
		chuanshutime+=chuanshutime1;
		}

		cudaEvent_t nstart5,nstop5;
		cudaEventCreate(&nstart5);
		cudaEventCreate(&nstop5);
		cudaEventRecord(nstart5);
		for( i=0;i<4;++i){
			offset = i*(ncol/4);
	   spcvmc_sw1<<<grid1,tBlock1,0,stream5[i]>>>(istart,iend,pbbcd,pbbcu,
		   pbbfd,pbbfu,pbbcddir,pbbfddir,puvcd,puvfd,puvcddir,puvfddir,
		   pnicd,pnifd,pnicddir,pnifddir,pnicu,pnifu,
		   zsflxzen,ztaug,ztaur,
		   ngc,ngs,bpade,exp_tbl,icpr,idelm,
		   iout,ptaucmc_d,lrtchkclr,zgcc,prmu0_d,ztauo,ztauc,zomcc,ztrac,ztradc,
		   zrefc,zrefdc,zincflx,adjflux,ztdbtc,ztdbtc_nodel,zdbtc,palbp_d,palbd_d,
		   ztdbt,ztdbt_nodel,zdbt,ztra,ztrad,zref,zrefd,lrtchkcld,pcldfmc_d,
		   ptaua_d,pomga_d,pasya_d,ptaormc_d,zomco,pomgcmc_d,zgco,pasycmc_d,zdbtc_nodel,
		   zrdndc,zrupc,zrupdc,zcu,zcd,zrdnd,zrup,zrupd,zfu,zfd,ztrao,
		   ztrado,zrefo,zrefdo,offset);
	   spcvmc_sw2<<<ceil(ncol*1.0/(tpb*8)),(tpb*8),0,stream5[i]>>>(istart,iend,pbbcd,pbbcu,
					   pbbfd,pbbfu,pbbcddir,pbbfddir,puvcd,puvfd,puvcddir,puvfddir,
					   pnicd,pnifd,pnicddir,pnifddir,pnicu,pnifu,
					   ngc,ngs,idelm,
					   iout,ztrac,ztradc,
					   zrefc,zrefdc,zincflx,ztdbtc,ztdbtc_nodel,zdbtc,
					   ztdbt,ztdbt_nodel,zdbt,ztra,ztrad,zref,zrefd,
					   zrdndc,zrupc,zrupdc,zcu,zcd,zrdnd,zrup,zrupd,zfu,zfd,offset);
	   }
						
		cudaEventRecord(nstop5);
		
		cudaEventSynchronize(nstop5);

		cudaEventElapsedTime(&elapsedTime5, nstart5, nstop5);
		spcvmctime+=elapsedTime5;

		for( i=0;i<4;++i){
			cudaEvent_t nstart333,nstop333;
			cudaEventCreate(&nstart333);
			cudaEventCreate(&nstop333);
			cudaEventRecord(nstart333);
			
			cudaMemcpyAsync(&zbbfd[i*(ncol*53/4)],&pbbfd[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&zbbfu[i*(ncol*53/4)],&pbbfu[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&zbbcd[i*(ncol*53/4)],&pbbcd[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&zbbcu[i*(ncol*53/4)],&pbbcu[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&zuvfd[i*(ncol*53/4)],&puvfd[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&zuvcd[i*(ncol*53/4)],&puvcd[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&znifd[i*(ncol*53/4)],&pnifd[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&znicd[i*(ncol*53/4)],&pnicd[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&znifu[i*(ncol*53/4)],&pnifu[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&znicu[i*(ncol*53/4)],&pnicu[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&zbbfddir[i*(ncol*53/4)],&pbbfddir[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&zbbcddir[i*(ncol*53/4)],&pbbcddir[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&zuvfddir[i*(ncol*53/4)],&puvfddir[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&zuvcddir[i*(ncol*53/4)],&puvcddir[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&znifddir[i*(ncol*53/4)],&pnifddir[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			cudaMemcpyAsync(&znicddir[i*(ncol*53/4)],&pnicddir[i*(ncol*53/4)],ncol*53*sizeof(double)/4,cudaMemcpyDeviceToHost,stream5[i]);
			
			cudaEventRecord(nstop333);
			
			cudaEventSynchronize(nstop333);
	
			cudaEventElapsedTime(&chuanshutime3, nstart333, nstop333);
			chuanshutime2+=chuanshutime3;
			}
		


		for( i=0;i<4;++i){
			cudaStreamDestroy(stream5[i]);
		}

			g = g + 1;
		printf("step=%d\n",g);
		
	}

	printf("zbbfddir=%E   %E\n",zbbfddir[29*ncol+191],zbbfddir[20*ncol+453]);
	printf("znifddir=%E   %E\n",znifddir[47*ncol+258],znifddir[31*ncol+478]);
	printf("znicddir=%E   %E\n",znicddir[32*ncol+369],znicddir[48*ncol+652]);
	printf("zuvfddir=%E   %E\n",zuvfddir[48*ncol+751],zuvfddir[4*ncol+365]);
	printf("zuvcddir=%E   %E\n",zuvcddir[42*ncol+388],zuvcddir[39*ncol+752]);
	printf("zbbcddir=%E   %E\n",zbbcddir[21*ncol+1011],zbbcddir[18*ncol+999]);
	printf("zbbfd=%E   %E\n",zbbfd[37*ncol+344],zbbfd[35*ncol+99]);
	printf("zbbfu=%E   %E\n",zbbfu[50*ncol+54],zbbfu[47*ncol+99]);
	printf("zbbcd=%E   %E\n",zbbcd[11*ncol+125],zbbcd[23*ncol+100]);
	printf("zbbcu=%E   %E\n",zbbcu[41*ncol+964],zbbcu[30*ncol+299]);
	printf("zuvfd=%E   %E\n",zuvfd[14*ncol+475],zuvfd[35*ncol+189]);
	printf("zuvcd=%E   %E\n",zuvcd[9*ncol+142],zuvcd[38*ncol+77]);
	printf("znifd=%E   %E\n",znifd[16*ncol+136],znifd[39*ncol+100]);
	printf("znicd=%E   %E\n",znicd[4*ncol+55],znicd[47*ncol+1000]);
	printf("znifu=%E   %E\n",znifu[46*ncol+9],znifu[48*ncol+699]);
	printf("znicu=%E   %E\n",znicu[1*ncol+0],znicu[23*ncol+179]);
	printf("\n\n");
		std::cout << "inatm time: " <<inatmtime<< std::endl;
		std::cout << "cldprmc time: " <<cldprmctime<< std::endl;
		std::cout << "setcoef time: " <<setcoeftime<< std::endl;
		std::cout << "taumol time: " <<taumoltime<< std::endl;
		std::cout << "spcvmc time: " <<spcvmctime<< std::endl;
		std::cout << "Host to Device time: " <<chuanshutime<< std::endl;
		std::cout << "Device to Host time: " <<chuanshutime2<< std::endl;

		cudaFree(play_d);
		cudaFree(plev_d);
		cudaFree(tlay_d);
		cudaFree(tlev_d);
		cudaFree(tsfc_d);
		cudaFree(h2ovmr_d);
		cudaFree(o3vmr_d);
		cudaFree(co2vmr_d);
		cudaFree(ch4vmr_d);
		cudaFree(o2vmr_d);
		cudaFree(n2ovmr_d);
		cudaFree(solvar_d);
		cudaFree(cldfmcl_d);
		cudaFree(taucmcl_d);
		cudaFree(ssacmcl_d);
		cudaFree(asmcmcl_d);
		cudaFree(fsfcmcl_d);
		cudaFree(ciwpmcl_d);
		cudaFree(clwpmcl_d);
		cudaFree(reicmcl_d);
		cudaFree(relqmcl_d);
		cudaFree(tauaer_d);
		cudaFree(ssaaer_d);
		cudaFree(asmaer_d);

		cudaFree(pavel);
		cudaFree(pz);
		cudaFree(pdp);
		cudaFree(tavel);
		cudaFree(tz);
		cudaFree(tbound);
		cudaFree(adjflux);
		cudaFree(wkl);
		cudaFree(coldry);
		cudaFree(cldfmc);
		cudaFree(taucmc);
		cudaFree(ssacmc);
		cudaFree(asmcmc);
		cudaFree(fsfcmc);
		cudaFree(ciwpmc);
		cudaFree(clwpmc);
		cudaFree(reicmc);
		cudaFree(dgesmc);
		cudaFree(relqmc);
		cudaFree(taua);
		cudaFree(ssaa);
		cudaFree(asma);

		cudaFree(taormc);

		cudaFree(extliq1);
		cudaFree(ssaliq1);
		cudaFree(asyliq1);
		cudaFree(extice2);
		cudaFree(ssaice2);
		cudaFree(asyice2);
		cudaFree(extice3);
		cudaFree(ssaice3);
		cudaFree(asyice3);
		cudaFree(fdlice3);
		cudaFree(abari);
		cudaFree(bbari);
		cudaFree(cbari);
		cudaFree(dbari);
		cudaFree(ebari);
		cudaFree(fbari);
		cudaFree(wavenum2);
		cudaFree(ngb);

		cudaFree(laytrop);
		cudaFree(layswtch);
		cudaFree(laylow);
		cudaFree(jp);
		cudaFree(jt);
		cudaFree(jt1);
		cudaFree(indself);
		cudaFree(indfor);
		cudaFree(colmol);
		cudaFree(co2mult);
		cudaFree(colh2o);
		cudaFree(colco2);
		cudaFree(colo3);
		cudaFree(coln2o);
		cudaFree(colch4);
		cudaFree(colo2);
		cudaFree(selffac);
		cudaFree(selffrac);
		cudaFree(forfac);
		cudaFree(forfrac);
		cudaFree(fac00);
		cudaFree(fac01);
		cudaFree(fac10);
		cudaFree(fac11);

		cudaFree(preflog);
		cudaFree(tref);

		cudaFree(palbd_d);
		cudaFree(palbp_d);
		cudaFree(prmu0_d);
		cudaFree(pcldfmc_d);
		cudaFree(ptaucmc_d);
		cudaFree(pasycmc_d);
		cudaFree(pomgcmc_d);
		cudaFree(ptaormc_d);
		cudaFree(ptaua_d);
		cudaFree(pasya_d);
		cudaFree(pomga_d);

		cudaFree(pbbcd);
		cudaFree(pbbcu);
		cudaFree(pbbfd);
		cudaFree(pbbfu);
		cudaFree(pbbfddir);
		cudaFree(pbbcddir);
		cudaFree(puvcd);
		cudaFree(puvfd);
		cudaFree(puvcddir);
		cudaFree(puvfddir);
		cudaFree(pnicd);
		cudaFree(pnifd);
		cudaFree(pnicddir);
		cudaFree(pnifddir);
		cudaFree(pnicu);
		cudaFree(pnifu);

		cudaFree(ngc);
		cudaFree(ngs);
		cudaFree(exp_tbl);

		cudaFree(zcd);
		cudaFree(zcu);
		cudaFree(zfd);
		cudaFree(zfu);

		cudaFree(zrefc);
		cudaFree(zrefdc);
		cudaFree(ztrac);
		cudaFree(ztradc);

		cudaFree(zgcc);
		cudaFree(ztauc);
		cudaFree(zomcc);
		cudaFree(lrtchkclr);
		cudaFree(lrtchkcld);

		cudaFree(ztaug);
		cudaFree(ztaur);
		cudaFree(zsflxzen);

		cudaFree(nspa);
		cudaFree(nspb);
		cudaFree(absa16);
		cudaFree(absb16);
		cudaFree(selfref16);
		cudaFree(forref16);
		cudaFree(sfluxref16);

		cudaFree(absa17);
		cudaFree(absb17);
		cudaFree(selfref17);
		cudaFree(forref17);
		cudaFree(sfluxref17);

		cudaFree(absa18);
		cudaFree(absb18);
		cudaFree(selfref18);
		cudaFree(forref18);
		cudaFree(sfluxref18);

		cudaFree(absa19);
		cudaFree(absb19);
		cudaFree(selfref19);
		cudaFree(forref19);
		cudaFree(sfluxref19);

		cudaFree(absa20);
		cudaFree(absb20);
		cudaFree(selfref20);
		cudaFree(forref20);
		cudaFree(sfluxref20);
		cudaFree(absch420);

		cudaFree(absa21);
		cudaFree(absb21);
		cudaFree(selfref21);
		cudaFree(forref21);
		cudaFree(sfluxref21);

		cudaFree(absa22);
		cudaFree(absb22);
		cudaFree(selfref22);
		cudaFree(forref22);
		cudaFree(sfluxref22);

		cudaFree(absa23);
		cudaFree(selfref23);
		cudaFree(forref23);
		cudaFree(sfluxref23);
		cudaFree(rayl23);

		cudaFree(absa24);
		cudaFree(absb24);
		cudaFree(selfref24);
		cudaFree(forref24);
		cudaFree(sfluxref24);
		cudaFree(abso3a24);
		cudaFree(abso3b24);
		cudaFree(rayla24);
		cudaFree(raylb24);

		cudaFree(absa25);
		cudaFree(sfluxref25);
		cudaFree(abso3a25);
		cudaFree(abso3b25);
		cudaFree(rayl25);

		cudaFree(sfluxref26);
		cudaFree(rayl26);

		cudaFree(absa27);
		cudaFree(absb27);
		cudaFree(sfluxref27);
		cudaFree(rayl27);

		cudaFree(absa28);
		cudaFree(absb28);
		cudaFree(sfluxref28);

		cudaFree(absa29);
		cudaFree(absb29);
		cudaFree(selfref29);
		cudaFree(forref29);
		cudaFree(sfluxref29);
		cudaFree(absh2o29);
		cudaFree(absco229);

		cudaFree(zomco);
		cudaFree(ztrao);
		cudaFree(zrefdo);
		cudaFree(zgco);
		cudaFree(zref);
		cudaFree(zrupd);
		cudaFree(zdbt);
		cudaFree(zrefo);
		cudaFree(zrup);
		cudaFree(ztrado);
		cudaFree(zrdnd);
		cudaFree(ztdbt);
		cudaFree(ztauo);
		cudaFree(zrupc);
		cudaFree(zrupdc);
		cudaFree(zrdndc);
		cudaFree(ztdbtc);
		cudaFree(zdbtc);
		cudaFree(zrefd);
		cudaFree(ztra);
		cudaFree(ztrad);
		cudaFree(zincflx);
		cudaFree(ztdbtc_nodel);
		cudaFree(ztdbt_nodel);

		cudaFreeHost(play);
		cudaFreeHost(plev);
		cudaFreeHost(tlay);
		cudaFreeHost(tsfc);
		cudaFreeHost(h2ovmr);
		cudaFreeHost(o3vmr);
		cudaFreeHost(co2vmr);
		cudaFreeHost(ch4vmr);
		cudaFreeHost(o2vmr);
		cudaFreeHost(n2ovmr);
		cudaFreeHost(solvar);
		cudaFreeHost(cldfmcl);
		cudaFreeHost(taucmcl);
		cudaFreeHost(ssacmcl);
		cudaFreeHost(asmcmcl);
		cudaFreeHost(fsfcmcl);
		cudaFreeHost(ciwpmcl);
		cudaFreeHost(clwpmcl);
		cudaFreeHost(reicmcl);
		cudaFreeHost(relqmcl);
		cudaFreeHost(tauaer);
		cudaFreeHost(ssaaer);
		cudaFreeHost(asmaer);

	//inatm out释放空间
		cudaFreeHost(pavel_c);
		cudaFreeHost(tavel_c);
		cudaFreeHost(pz_c);
		cudaFreeHost(tz_c);
		cudaFreeHost(tbound_c);
		cudaFreeHost(pdp_c);
		cudaFreeHost(coldry_c);
		cudaFreeHost(wkl_c);
		cudaFreeHost(adjflux_c);
		cudaFreeHost(taua_c);
		cudaFreeHost(ssaa_c);
		cudaFreeHost(asma_c);
		cudaFreeHost(cldfmc_c);
		cudaFreeHost(taucmc_c);
		cudaFreeHost(ssacmc_c);
		cudaFreeHost(asmcmc_c);
		cudaFreeHost(fsfcmc_c);
		cudaFreeHost(ciwpmc_c);
		cudaFreeHost(clwpmc_c);
		cudaFreeHost(reicmc_c);
		cudaFreeHost(dgesmc_c);
		cudaFreeHost(relqmc_c);

		cudaFreeHost(taormc_c);
		cudaFreeHost(extliq1_c);
		cudaFreeHost(ssaliq1_c);
		cudaFreeHost(asyliq1_c);
		cudaFreeHost(extice2_c);
		cudaFreeHost(ssaice2_c);
		cudaFreeHost(asyice2_c);
		cudaFreeHost(extice3_c);
		cudaFreeHost(ssaice3_c);
		cudaFreeHost(asyice3_c);
		cudaFreeHost(fdlice3_c);
		cudaFreeHost(abari_c);
		cudaFreeHost(bbari_c);
		cudaFreeHost(cbari_c);
		cudaFreeHost(dbari_c);
		cudaFreeHost(ebari_c);
		cudaFreeHost(fbari_c);
		cudaFreeHost(wavenum2_c);
		cudaFreeHost(ngb_c);

		cudaFreeHost(laytrop_c);
		cudaFreeHost(layswtch_c);
		cudaFreeHost(laylow_c);
		cudaFreeHost(jp_c);
		cudaFreeHost(jt_c);
		cudaFreeHost(jt1_c);
		cudaFreeHost(indself_c);
		cudaFreeHost(indfor_c);
		cudaFreeHost(colmol_c);
		cudaFreeHost(co2mult_c);
		cudaFreeHost(colh2o_c);
		cudaFreeHost(colco2_c);
		cudaFreeHost(colo3_c);
		cudaFreeHost(coln2o_c);
		cudaFreeHost(colch4_c);
		cudaFreeHost(colo2_c);
		cudaFreeHost(selffac_c);
		cudaFreeHost(selffrac_c);
		cudaFreeHost(forfac_c);
		cudaFreeHost(forfrac_c);
		cudaFreeHost(fac00_c);
		cudaFreeHost(fac01_c);
		cudaFreeHost(fac10_c);
		cudaFreeHost(fac11_c);

		cudaFreeHost(preflog_c);
		cudaFreeHost(tref_c);

		cudaFreeHost(albdif);
		cudaFreeHost(albdir);
		cudaFreeHost(cossza);
		cudaFreeHost(zcldfmc);
		cudaFreeHost(ztaucmc);
		cudaFreeHost(zasycmc);
		cudaFreeHost(zomgcmc);
		cudaFreeHost(ztaormc);
		cudaFreeHost(ztaua);
		cudaFreeHost(zasya);
		cudaFreeHost(zomga);

		cudaFreeHost(ngs_c);
		cudaFreeHost(ngc_c);
		cudaFreeHost(exp_tbl_c);

		cudaFreeHost(absa16_c);
		cudaFreeHost(absb16_c);
		cudaFreeHost(selfref16_c);
		cudaFreeHost(forref16_c);
		cudaFreeHost(sfluxref16_c);

		cudaFreeHost(absa17_c);
		cudaFreeHost(absb17_c);
		cudaFreeHost(selfref17_c);
		cudaFreeHost(forref17_c);
		cudaFreeHost(sfluxref17_c);

		cudaFreeHost(absa18_c);
		cudaFreeHost(absb18_c);
		cudaFreeHost(selfref18_c);
		cudaFreeHost(forref18_c);
		cudaFreeHost(sfluxref18_c);

		cudaFreeHost(absa19_c);
		cudaFreeHost(absb19_c);
		cudaFreeHost(selfref19_c);
		cudaFreeHost(forref19_c);
		cudaFreeHost(sfluxref19_c);

		cudaFreeHost(absa20_c);
		cudaFreeHost(absb20_c);
		cudaFreeHost(selfref20_c);
		cudaFreeHost(forref20_c);
		cudaFreeHost(sfluxref20_c);
		cudaFreeHost(absch420_c);

		cudaFreeHost(absa21_c);
		cudaFreeHost(absb21_c);
		cudaFreeHost(selfref21_c);
		cudaFreeHost(forref21_c);
		cudaFreeHost(sfluxref21_c);

		cudaFreeHost(absa22_c);
		cudaFreeHost(absb22_c);
		cudaFreeHost(selfref22_c);
		cudaFreeHost(forref22_c);
		cudaFreeHost(sfluxref22_c);

		cudaFreeHost(absa23_c);
		cudaFreeHost(selfref23_c);
		cudaFreeHost(forref23_c);
		cudaFreeHost(sfluxref23_c);
		cudaFreeHost(rayl23_c);

		cudaFreeHost(absa24_c);
		cudaFreeHost(absb24_c);
		cudaFreeHost(selfref24_c);
		cudaFreeHost(forref24_c);
		cudaFreeHost(sfluxref24_c);
		cudaFreeHost(abso3a24_c);
		cudaFreeHost(abso3b24_c);
		cudaFreeHost(rayla24_c);
		cudaFreeHost(raylb24_c);

		cudaFreeHost(absa25_c);
		cudaFreeHost(sfluxref25_c);
		cudaFreeHost(abso3a25_c);
		cudaFreeHost(abso3b25_c);
		cudaFreeHost(rayl25_c);

		cudaFreeHost(sfluxref26_c);
		cudaFreeHost(rayl26_c);

		cudaFreeHost(absa27_c);
		cudaFreeHost(absb27_c);
		cudaFreeHost(sfluxref27_c);
		cudaFreeHost(rayl27_c);

		cudaFreeHost(absa28_c);
		cudaFreeHost(absb28_c);
		cudaFreeHost(sfluxref28_c);

		cudaFreeHost(absa29_c);
		cudaFreeHost(absb29_c);
		cudaFreeHost(selfref29_c);
		cudaFreeHost(forref29_c);
		cudaFreeHost(sfluxref29_c);
		cudaFreeHost(absh2o29_c);
		cudaFreeHost(absco229_c);

		cudaFreeHost(zsflxzen_c);
		cudaFreeHost(ztaug_c);
		cudaFreeHost(ztaur_c);

		cudaFreeHost(lrtchkclr_c);
		cudaFreeHost(lrtchkcld_c);

		cudaFreeHost(ztauc_c);
		cudaFreeHost(zomcc_c);
		cudaFreeHost(zgcc_c);

		cudaFreeHost(zbbcd);
		cudaFreeHost(zbbcu);
		cudaFreeHost(zbbfd);
		cudaFreeHost(zbbfu);
		cudaFreeHost(zbbfddir);
		cudaFreeHost(zbbcddir);
		cudaFreeHost(zuvcd);
		cudaFreeHost(zuvfd);
		cudaFreeHost(zuvcddir);
		cudaFreeHost(zuvfddir);
		cudaFreeHost(znicd);
		cudaFreeHost(znifd);
		cudaFreeHost(znicddir);
		cudaFreeHost(znifddir);
		cudaFreeHost(znicu);
		cudaFreeHost(znifu);

		cudaFreeHost(nspa_c);
		cudaFreeHost(nspb_c);
}

