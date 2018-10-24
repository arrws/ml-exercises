
# C++ Basic neural network


#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <cstdlib>
using namespace std;
#define HIGH_NUM 999999
#define LOW_NUM -999999
ifstream in("data.txt");
ofstream out("report.txt");

double eta=0.2; // Learning Rate [0-1]:  controls how much the weights are adjusted at each update
double alpha=0.5; // Momentum [0-1]:  keep the weight changes moving in a consistent direction

class Neuron;
typedef vector <Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned outputs,int index)
    {
        m_index=index;
        for (unsigned i=0;i<outputs;++i)
        {
            m_weights.push_back(rand()/double(RAND_MAX)); // initialized with random values
            m_delta_weights.push_back(0);
        }
    }
    void feed_layer(Layer &layer) // get output from previous layer
    {
        double s=0.0;
        for (unsigned i=0;i<layer.size();++i)
            s+=layer[i].getOutput()*layer[i].m_weights[m_index];
        m_output=fun(s); // apply the activation function
    }
    void update_weight(Layer &layer)
    {
        for(unsigned i=0;i<layer.size();++i)
        {
            double delta=eta*layer[i].getOutput()*m_gradient+alpha*layer[i].getDeltaWeight(m_index); // magic formula
            layer[i].setWeight(m_index,layer[i].getWeight(m_index)+delta); // update weight
            layer[i].setDeltaWeight(m_index,delta); // update weight delta
        }
    }
    double fun(double x){ return tanh(x); }
    double fun_derivate(double x){ return 1.0-x*x; }
    double sum(Layer &layer)
    {
        double s=0.0;
        for(unsigned i=0;i<layer.size()-1;++i)
            s+=m_weights[i]*layer[i].getGradient();
        return s;
    }
    void setOutput(double val){ m_output=val;}
    void setGradient(double x){ m_gradient=x*fun_derivate(m_output); } // compute the gradient
    void setWeight(unsigned index, double x){ m_weights[index]=x; }
    void setDeltaWeight(unsigned index, double x){ m_delta_weights[index]=x; }
    double getOutput(){ return m_output;}
    double getGradient(){ return m_gradient; }
    double getWeight(unsigned index){ return m_weights[index]; }
    double getDeltaWeight(unsigned index){ return m_delta_weights[index]; }

private:
    int m_index; // neuron number in its layer
    double m_gradient,m_output;
    vector <double> m_weights,m_delta_weights;
};


class Net
{
public:
    Net(){}
    void build(vector <unsigned> &top)
    {
        unsigned layers=top.size();
        for(unsigned i=0;i<layers;++i)
        {
            m_layers.push_back(Layer()); // add layer
            int outputs=0;
            if(i<layers-1) // last layer's neurons don't have outputs
                outputs=top[i+1];
            for(unsigned j=0;j<=top[i];++j) // populate layer
                m_layers.back().push_back(Neuron(outputs,j));
            m_layers.back().back().setOutput(1.0); // add bias neuron
        }
    }
    void destroy(){ m_layers.clear(); } // empty the net
    void feed_forward(vector <double> &v_in)
    {
        // assert inputVals.size() == m_layers[0].size()-1
        // set output values for output layer
        for(unsigned j=0;j<v_in.size();++j)
            m_layers[0][j].setOutput(v_in[j]);
        // set output values for hidden layers
        for(unsigned i=1;i<m_layers.size();++i)
            for(unsigned j=0;j<m_layers[i].size()-1;++j)
                m_layers[i][j].feed_layer(m_layers[i-1]);
    }
    void back_propagate(vector <double> &v_out) // update weights
    {
        // set gradient for output layer
        Layer &layer=m_layers.back();
        for (unsigned i=0;i<layer.size()-1;++i){

            layer[i].setGradient(v_out[i]-layer[i].getOutput());}
        // set gradient for hidden layer
        for(unsigned j=m_layers.size()-2;j>0;--j)
            for (unsigned i=0;i<m_layers[j].size();++i)
                m_layers[j][i].setGradient(m_layers[j][i].sum(m_layers[j+1]));
        // update weights
        for(unsigned j=m_layers.size()-1;j>0;--j)
            for (unsigned i=0;i<m_layers[j].size()-1;++i)
                m_layers[j][i].update_weight(m_layers[j-1]);
    }
    void get_results(vector <double> &v_res) // return the values from the last layer
    {
        for (unsigned i=0;i<m_layers.back().size()-1;++i){
            v_res.push_back(m_layers.back()[i].getOutput());}
    }
private:
    vector <Layer> m_layers; // structure that contains all layers
};

double norm(double x, double xmin, double xmax)
{
    return (x-xmin)/(xmax-xmin); // the normalization formula
}

void get_config(vector <unsigned> &top) // get the neuronet topology
{
    cout<<"\n----------------> CONFIGURE:\n";
    unsigned n,x;
    in.close();
    in.open("data.txt");
    in>>x;
    top.push_back(x);
    cout<<"input neurons: "<<x<<"\n";
    cout<<"number of layers: ";
    cin>>n;
    for(unsigned i=1;i<=n;++i)
    {
        cout<<"layer "<<i<<" neurons: ";
        cin>>x;
        top.push_back(x);
    }
    in>>x;
    top.push_back(x);
    cout<<"output neurons: "<<x<<"\n";
}

void get_max_min(unsigned n,vector <double> &min_val,vector <double> &max_val)
{
    // compute min max values for every set of values
    unsigned m;
    double x;
    in>>m;
    while(m--)
    {
        for (unsigned i=0;i<n;++i)
        {
            in>>x;
            if(max_val[i]<x)
                max_val[i]=x;
            if(min_val[i]>x)
                min_val[i]=x;
        }
    }
}

void print_report(vector <double> &v_res,vector <double> &v_out)
{
    // print all results
    out<<"\nresult: ";
    for(unsigned i=0;i<v_res.size();++i)
        out<<v_res[i]<<' ';
    out<<"\ntarget: ";
    for(unsigned i=0;i<v_out.size();++i)
        out<<v_out[i]<<' ';
}

int main()
{
    unsigned m,n,input,output;
    in>>input>>output;
    n=input+output;
    vector<double> min_val(n,HIGH_NUM),max_val(n,LOW_NUM);
    get_max_min(n,min_val,max_val);

    vector <double> v_in,v_out,v_res; // containers for input, target-output and result values
    vector <unsigned> top; // topology

    bool finish=false;
    char answer;
    Net net;
    while(!finish)
    {
        get_config(top);
        cout<<"\n---> Start? y/n\n";
        cin>>answer;
        if(answer=='y')
        {
            out.close();
            out.open("report.txt");
            double x,error=0;
            net.build(top);

            in>>m;
            while(m--)
            {
                // get target-output values
                for (unsigned i=0;i<top.back();++i){
                    in>>x;
                    v_out.push_back(norm(x,min_val[i],max_val[i])); // add normalized value
                }
                // get input values
                for (unsigned i=top.back();i<top.back()+top.front();++i){
                    in>>x;
                    v_in.push_back(norm(x,min_val[i],max_val[i])); // add normalized value
                }

                net.feed_forward(v_in);
                net.get_results(v_res);
                net.back_propagate(v_out);
                print_report(v_res,v_out);

                // error progress
                for(unsigned i=0;i<v_res.size();++i)
                    error+=abs(v_res[i]-v_out[i]);
                if(m%50==0)
                {
                    cout<<"error: "<<error/50<<"\n";
                    error=0;
                }
                // Clear Containers
                v_in.clear();
                v_out.clear();
                v_res.clear();
            }
            net.destroy();
            top.clear();
        }
        cout<<"\n---> Try Again? y/n\n";
        cin>>answer;
        if(answer!='y')
            finish=true;
    }
    cout<<"Bye!";
    return 0;
}
