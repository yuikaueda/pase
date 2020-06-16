#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#include<array>
#include<string>
#include<fstream>


#define nx  32 //# number of computational grids along x direction
#define ny  32 //# number of computational grids along y direction
#define dx  2.0e-9 //# spacing of computational grids [m]
#define dy  2.0e-9
#define c0  0.5 //# average composition of B atom [atomic fraction]
#define R   8.314 //# gas constant
#define temp  673  //temperature [K]
#define nsteps  600 //total number of time-steps


double Uniform(){
  return ((double)rand())/((double)RAND_MAX+1);
}


void update_orderparameter(std::array<std::array<double, ny>, nx> c)
{

  std::array<std::array<double, ny>, nx> c_new;

  double La = 20000 - 9*temp;
  double ac = 3.0e-14; //# gradient coefficient [Jm2/mol]
  double Da = 1.0e-04*exp(-300000.0/R/temp); //# diffusion coefficient of A atom [m2/s]
  double Db = 2.0e-05*exp(-300000.0/R/temp); //# diffusion coefficient of B atom [m2/s]
  double dt = (dx*dx/Da)*0.1; //# time increment [s]
  //double c_new[nx][ny]

  for (int j = 0; j < ny; j++){
    for (int i = 0; i < nx; i++){
      int ip = i + 1;
      int im = i - 1;
      int jp = j + 1;
      int jm = j - 1;
      int ipp = i + 2;
      int imm = i - 2;
      int jpp = j + 2;
      int jmm = j - 2;

      if (ip > nx -1){    //pediodic boundary condition
        ip = ip - nx;
      }
      
      if (im < 0){
        im = im + nx;
      }
      
     if (jp > ny - 1){
        jp = jp - ny;
     }

     if (jm < 0){
        jm = jm + ny;
     }

     if (ipp > nx - 1){
        ipp = ipp - nx;
     }

     if (imm < 0){
        imm = imm + nx;
     }

     if (jpp > ny -1){
        jpp = jpp -ny;
     }

     if (jmm < 0){
        jmm = jmm + ny;
     }
         
     double cc = c[i][j]; //# at (i,j) "centeral point"
     double ce = c[ip][j]; //# at (i+1.j) "eastern point"
     double cw = c[im][j]; //# at (i-1,j) "western point"
     double cs = c[i][jm]; //# at (i,j-1) "southern point"
     double cn = c[i][jp]; //# at (i,j+1) "northern point"
     double cse = c[ip][jm]; //# at (i+1, j-1)
     double cne = c[ip][jp];
     double csw = c[im][jm];
     double cnw = c[im][jp];
     double cee = c[ipp][j];  //# at (i+2, j+1)
     double cww = c[imm][j];
     double css = c[i][jmm];
     double cnn = c[i][jpp];

     double mu_chem_c = R*temp*(log(cc)-log(1.0-cc)) + La*(1.0-2.0*cc); //# chemical term of the diffusion potential
     double mu_chem_w = R*temp*(log(cw)-log(1.0-cw)) + La*(1.0-2.0*cw); 
     double mu_chem_e = R*temp*(log(ce)-log(1.0-ce)) + La*(1.0-2.0*ce); 
     double mu_chem_n = R*temp*(log(cn)-log(1.0-cn)) + La*(1.0-2.0*cn);  
     double mu_chem_s = R*temp*(log(cs)-log(1.0-cs)) + La*(1.0-2.0*cs); 
                                                                                                                   
     double mu_grad_c = -ac*( (ce -2.0*cc +cw )/dx/dx + (cn  -2.0*cc +cs )/dy/dy ); //# gradient term of the diffusion potential
     double mu_grad_w = -ac*( (cc -2.0*cw +cww)/dx/dx + (cnw -2.0*cw +csw)/dy/dy );
     double mu_grad_e = -ac*( (cee-2.0*ce +cc )/dx/dx + (cne -2.0*ce +cse)/dy/dy ); 
     double mu_grad_n = -ac*( (cne-2.0*cn +cnw)/dx/dx + (cnn -2.0*cn +cc )/dy/dy ); 
     double mu_grad_s = -ac*( (cse-2.0*cs +csw)/dx/dx + (cc  -2.0*cs +css)/dy/dy ); 

     double mu_c = mu_chem_c + mu_grad_c; //# total diffusion potental
     double mu_w = mu_chem_w + mu_grad_w; 
     double mu_e = mu_chem_e + mu_grad_e; 
     double mu_n = mu_chem_n + mu_grad_n; 
     double mu_s = mu_chem_s + mu_grad_s; 
                                                                               
     double nabla_mu = (mu_w -2.0*mu_c + mu_e)/dx/dx + (mu_n -2.0*mu_c + mu_s)/dy/dy;    
     double dc2dx2 = ((ce-cw)*(mu_e-mu_w))/(4.0*dx*dx);
     double dc2dy2 = ((cn-cs)*(mu_n-mu_s))/(4.0*dy*dy); 
  
     double DbDa = Db/Da;
     double mob = (Da/R/temp)*(cc+DbDa*(1.0-cc))*cc*(1.0-cc); 
     double dmdc = (Da/R/temp)*((1.0-DbDa)*cc*(1.0-cc)+(cc+DbDa*(1.0-cc))*(1.0-2.0*cc)); 

     double dcdt = mob*nabla_mu + dmdc*(dc2dx2 + dc2dy2); //# right-hand side of Cahn-Hilliard equation
     c_new[i][j] = c[i][j] + dcdt *dt; //# update order parameter c
    }
  }

  c = c_new;
  
}

int main()
{

  std::array<std::array<double, ny>, nx> c_;

  for(int j = 0; j < c_.size(); j++){
    for( int i = 0; i < c_[0].size(); i++){
       c_[i][j] = 0;
    }
  }
//  std::array<std::array<double, ny>, nx> cnew_;

//  double c_[nx][ny] = {0};
//  double cnew_[nx][ny] = {0};

  srand ((unsigned)time(NULL));
  
  FILE* fp0;
  fp0 = fopen("ini_c.dat" , "w");
  if(fp0==NULL){
  printf("File open faild.");
  }

  for (int j = 0; j < ny; j++){
    for (int i = 0; i < nx; i++){
      double num = Uniform();
 
      c_[i][j] += num;
  

    fprintf(fp0, "%2d\t%2d\t%f\n",(int)i , (int)j , c_[i][j]);

    }    
  }
  fclose(fp0);

for (int t =0; t < nsteps+ 1; t++){

     update_orderparameter(c_);
     //c[:,:] = c_new[:,:] //# swap c at time t and c at time t+dt
 

    if(t%600 == 0){
      std::ofstream fp;
      std::string filename;
      filename = "result"+std::to_string(t)+".dat";
      fp.open(filename, std::ios::out);

      for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
          fp << i << " " << j << " " << c_[i][j] << std::endl;
        }
      }
      fp.close();
    }  
}
  return 0;
}

