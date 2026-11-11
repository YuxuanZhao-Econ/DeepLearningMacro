using LinearAlgebra, Statistics, Random, Serialization, Printf, Flux, CUDA, Plots
BLAS.set_num_threads(1)
ENV["GKSwstype"]="100"
const HERE=@__DIR__
const ROOT=normpath(joinpath(HERE,"../.."))
const OUT=joinpath(HERE,"results")
mkpath(OUT)
module S
include(normpath(joinpath(@__DIR__,"../../../sequence_space_jacobian/src/ks1998/KS1998.jl")))
end
const p=S.KSParams(beta=.96,eis=1.,delta=.08,alpha=.36,L=1.,rho_e=.95,sd_e=.20,nE=3,nA=80,amax=60.)
const rho=.9
const sig=.01
const shock=.001
const T=120
price(K,z)=(p.alpha*z*K^(p.alpha-1)-p.delta,(1-p.alpha)*z*K^p.alpha)
println("Solving shared deterministic steady state")
const ss=S.compute_ks_steady_state(p)
const a=ss.a_grid; const e=ss.e_grid; const Pi=ss.Pi
const E=length(e); const A=length(a); const N=E*A
const av=vec(repeat(a',E,1)); const ev=vec(repeat(e,1,A))
println("SS K=$(ss.K), C=$(ss.C), market residual=$(ss.asset_mkt), top mass=$(sum(ss.D[:,end]))")
serialize(joinpath(OUT,"steady_state.jls"),(p=p,ss=ss))

# Sequence-space residual: Knext implied by backward household decisions and
# forward distribution transport, with inherited K[1] fixed.
function seq(Knext,z)
    K=vcat(ss.K,Knext[1:end-1]); prices=price.(K,z)
    hh=S.backward_policy_path(first.(prices),last.(prices),ss,p)
    D=copy(ss.D); implied=zeros(length(z)); C=similar(implied)
    Ds=Matrix{Float64}[]
    for t in eachindex(z)
        push!(Ds,copy(D)); C[t]=sum(D.*hh.c_policies[t])
        implied[t]=sum(D.*hh.a_policies[t])
        D=S.forward_distribution(D,hh.a_policies[t],a,Pi)
    end
    return (res=Knext-implied,K=K,C=C,Y=z.*K.^p.alpha,D=Ds)
end
function ssj()
    k=fill(ss.K,T); z=ones(T); h=1e-4
    J=zeros(T,T); B=zeros(T,T)
    for j=1:T
        d=zeros(T); d[j]=h
        J[:,j]=(seq(k+d,z).res-seq(k-d,z).res)/(2h)
        B[:,j]=(seq(k,z+d).res-seq(k,z-d).res)/(2h)
    end
    dz=shock*rho.^(0:T-1); dk=-(J\(B*dz))
    plus=seq(k+dk,z+dz); minus=seq(k-dk,z-dz)
    response=(K=(plus.K-minus.K)/2,C=(plus.C-minus.C)/2,Y=(plus.Y-minus.Y)/2,
        D=[(plus.D[t]-minus.D[t])/2 for t=1:T])
    println("SSJ residual $(maximum(abs.(J*dk+B*dz)))")
    return response
end

# Original KS algorithm adapted to continuous log-TFP AR(1), with Gaussian
# quadrature and a log-linear PLM. Household solution remains Euler/EGM.
const kg=collect(range(.7ss.K,1.3ss.K,length=11))
const zg=collect(range(-.10,.10,length=11))
const epsq=[-sqrt(3.),0.,sqrt(3.)]; const pq=[1/6,2/3,1/6]
function bracket(g,x)
    j=clamp(searchsortedlast(g,x),1,length(g)-1)
    return j,clamp((x-g[j])/(g[j+1]-g[j]),0.,1.)
end
function at(c,K,z)
    j,u=bracket(kg,K); l,v=bracket(zg,log(z))
    return (1-u)*(1-v)*c[:,:,j,l]+u*(1-v)*c[:,:,j+1,l]+(1-u)*v*c[:,:,j,l+1]+u*v*c[:,:,j+1,l+1]
end
function ks_house(coeff,c=nothing)
    if isnothing(c)
        c=repeat(ss.c_policy,1,1,length(kg),length(zg))
    end
    err=Inf
    for it=1:1200
        cn=similar(c)
        for l in eachindex(zg), j in eachindex(kg)
            K=kg[j]; z=exp(zg[l]); r,w=price(K,z)
            kp=exp(coeff[1]+coeff[2]*log(K)+coeff[3]*log(z))
            Em=zeros(E,A)
            for q=1:3
                zp=exp(rho*log(z)+sig*epsq[q]); rp,wp=price(kp,zp)
                Em .+= pq[q]*(1+rp).*(Pi*(1 ./ at(c,kp,zp)))
            end
            for ie=1:E
                ce=1 ./ (p.beta.*Em[ie,:]); xe=(ce+a.-w*e[ie])./(1+r)
                ap=[clamp(S.interp_linear(xe,a,x),0.,p.amax) for x in a]
                cn[ie,:,j,l]=(1+r).*a.+w*e[ie].-ap
            end
        end
        err=maximum(abs.(log.(cn).-log.(c))); c=cn
        err<1e-7 && break
    end
    return c,err
end
function ks_policy(c,D,z)
    K=sum(D.*a'); r,w=price(K,z)
    coh=(1+r).*repeat(a',E,1).+w.*repeat(e,1,A)
    cp=at(c,K,z); ap=clamp.(coh.-cp,0.,p.amax)
    return ap,coh-ap
end
function solveks()
    coeff=[.04log(ss.K),.96,.05]; c=nothing; err=Inf; hherr=Inf
    for it=1:60
        c,hherr=ks_house(coeff,c); D=copy(ss.D); z=1.; rng=MersenneTwister(777)
        X=zeros(3500,3); y=zeros(3500)
        for t=1:4000
            K=sum(D.*a'); ap,cp=ks_policy(c,D,z); kp=sum(D.*ap)
            if t>500
                X[t-500,:]=[1,log(K),log(z)]; y[t-500]=log(kp)
            end
            D=S.forward_distribution(D,ap,a,Pi); z=exp(rho*log(z)+sig*randn(rng))
        end
        fit=X\y; err=maximum(abs.(X*(fit-coeff)))
        r2=1-sum(abs2,y-X*fit)/sum(abs2,y.-mean(y))
        println("KS outer=$it forecast change=$err HH=$hherr R2=$r2")
        if err<2e-6
            return (c=c,coeff=coeff,fit=fit,error=err,hherror=hherr,r2=r2)
        end
        coeff=.5coeff+.5fit
    end
    return (c=c,coeff=coeff,error=err,hherror=hherr,r2=NaN)
end
function ks_path(c,z)
    D=copy(ss.D); Ks=zeros(T); Cs=zeros(T); Ds=Matrix{Float64}[]
    for t=1:T
        Ks[t]=sum(D.*a'); push!(Ds,copy(D)); ap,cp=ks_policy(c,D,z[t]); Cs[t]=sum(D.*cp)
        D=S.forward_distribution(D,ap,a,Pi)
    end
    return (K=Ks,C=Cs,Y=z.*Ks.^p.alpha,D=Ds)
end

# Full-distribution neural Euler solver. Raw histogram weights replace the
# original finite-particle raw input, so all three methods share the same
# income process, grid and exact distribution transport. Two hidden layers,
# sigmoid consumption share, positive multiplier and K=4 unbiased Euler U.
const gpu=CUDA.functional()
gpu && CUDA.allowscalar(false)
dev(x)=gpu ? CuArray(x) : x
const va=dev(av); const ve=dev(ev); const TP=dev(Pi)
const nodes=dev(reshape(a,:,1))
const left=dev(reshape(vcat(a[1]-1,a[1:end-1]),:,1))
const right=dev(reshape(vcat(a[2:end],a[end]+1),:,1))
const probs=[dev(vec(repeat(Pi[:,j],1,A))) for j=1:E]
function transport(D,ap)
    masses=[begin
        x=reshape(ap[i:E:N],1,:)
        weights=max.(0.,min.((x.-left)./(nodes.-left),(right.-x)./(right.-nodes)))
        reshape(weights*D[i:E:N],1,:)
    end for i=1:E]
    return vec(transpose(TP)*vcat(masses...))
end
function nnpolicy(net,D,z,coh,ee)
    x=vcat(repeat(reshape(100 .* D,:,1),1,length(coh)),
        reshape(coh./ss.K,1,:),reshape(ee,1,:),fill(log(z)/.05,1,length(coh)) |> dev)
    raw=net(x)
    xi=Flux.sigmoid.(vec(raw[1:1,:]).-1.5)
    ap=min.((1 .-xi).*coh,p.amax)
    return (ap=ap,c=max.(coh.-ap,1e-10),mu=exp.(clamp.(vec(raw[2:2,:]),-10.,10.)))
end
function nncurrent(net,D,z)
    K=sum(D.*va); r,w=price(K,z)
    return nnpolicy(net,D,z,(1+r).*va.+w.*ve,ve)
end
function objective(net,D,z,zps)
    now=nncurrent(net,D,z); Dp=transport(D,now.ap); kp=sum(D.*now.ap)
    es=[begin
        rp,wp=price(kp,zp)
        ecs=[begin
            ee=zero.(ve).+e[j]
            future=nnpolicy(net,Dp,zp,(1+rp).*now.ap.+wp.*ee,ee)
            probs[j]./future.c
        end for j=1:E]
        p.beta.*now.c.*(1+rp).*reduce(+,ecs).-now.mu
    end for zp in zps]
    su=reduce(+,es); sq=reduce(+,[x.^2 for x in es])
    b=now.ap./now.c; m=now.mu.-1
    fb=m.+b.-sqrt.(m.^2.+b.^2.+1e-20)
    loss=sum(D.*(fb.^2.+(su.^2.-sq)./12))
    return loss,(Dp=Dp,fb=sum(D.*fb.^2))
end
function trainnn()
    Random.seed!(1234)
    net=Flux.f64(Chain(Dense(N+3=>64,Flux.sigmoid),Dense(64=>64,Flux.sigmoid),Dense(64=>2)))
    net=gpu ? Flux.gpu(net) : net
    opt=Flux.setup(Flux.Adam(1e-3),net); D=dev(vec(ss.D)); z=1.; rng=MersenneTwister(2026)
    losses=Float64[]
    for t=1:20000
        zps=exp.(rho*log(z).+sig.*randn(rng,4))
        val,gr=Flux.withgradient(net) do model
            objective(model,D,z,zps)
        end
        isfinite(val[1]) || error("NN nonfinite loss")
        Flux.update!(opt,net,gr[1]); D=val[2].Dp; z=zps[1]
        push!(losses,val[1])
        if t%1000==0
            println("NN step=$t loss=$(mean(losses[end-999:end])) mass=$(sum(D)) K=$(sum(D.*va))")
            serialize(joinpath(OUT,"nn_checkpoint.jls"),(net=Flux.cpu(net),losses=losses,steps=t))
        end
        t==12000 && Flux.adjust!(opt,3e-4)
        t%100==0 && (D=dev(vec(ss.D)); z=exp(.02randn(rng)))
    end
    return Flux.cpu(net)
end
function nnpath(net,z)
    net=gpu ? Flux.gpu(net) : net; D=dev(vec(ss.D)); Ks=zeros(T); Cs=zeros(T); Ds=Matrix{Float64}[]
    for t=1:T
        Ks[t]=sum(D.*va); push!(Ds,reshape(Array(D),E,A)); now=nncurrent(net,D,z[t]); Cs[t]=sum(D.*now.c)
        D=transport(D,now.ap)
    end
    return (K=Ks,C=Cs,Y=z.*Ks.^p.alpha,D=Ds)
end
diffpath(x,y)=(K=x.K-y.K,C=x.C-y.C,Y=x.Y-y.Y,D=[x.D[t]-y.D[t] for t=1:T])
function main()
    sr=isfile(joinpath(OUT,"ssj.jls")) ? deserialize(joinpath(OUT,"ssj.jls")) : ssj()
    serialize(joinpath(OUT,"ssj.jls"),sr)
    ks=isfile(joinpath(OUT,"ks.jls")) ? deserialize(joinpath(OUT,"ks.jls")) : solveks()
    serialize(joinpath(OUT,"ks.jls"),ks)
    net=isfile(joinpath(OUT,"nn_final.jls")) ? deserialize(joinpath(OUT,"nn_final.jls")) : trainnn()
    serialize(joinpath(OUT,"nn_final.jls"),net)
    zb=ones(T); zs=exp.(shock.*rho.^(0:T-1))
    kb=ks_path(ks.c,zb); kr=diffpath(ks_path(ks.c,zs),kb)
    nb=nnpath(net,zb); nr=diffpath(nnpath(net,zs),nb)
    responses=[sr,kr,nr]; names=["SSJ linear","KS PLM","Neural Euler"]
    open(joinpath(OUT,"irfs.csv"),"w") do io
        println(io,"method,horizon,dK,dC,dY,W1_shock_baseline,TV_shock_baseline,mass_response")
        for (name,r) in zip(names,responses),t=1:T
            marginal=vec(sum(r.D[t],dims=1)); w1=sum(abs.(cumsum(marginal)[1:end-1]).*diff(a))
            println(io,join((name,t-1,r.K[t],r.C[t],r.Y[t],w1,.5sum(abs,r.D[t]),sum(r.D[t])),","))
        end
    end
    panels=[]
    for (key,denom) in ((:Y,ss.Y),(:C,ss.C),(:K,ss.K))
        fig=plot(xlabel="period",ylabel="% of common steady state",title=string(key))
        for (name,r) in zip(names,responses)
            plot!(fig,0:59,100getproperty(r,key)[1:60]./denom,label=name,lw=2)
        end
        push!(panels,fig)
    end
    fig=plot(xlabel="period",ylabel="asset units",title="Wasserstein-1: shock vs baseline")
    for (name,r) in zip(names,responses)
        w1=[sum(abs.(cumsum(vec(sum(d,dims=1))))[1:end-1].*diff(a)) for d in r.D]
        plot!(fig,0:59,w1[1:60],label=name,lw=2)
    end
    push!(panels,fig)
    savefig(plot(panels...,layout=(2,2),size=(1100,750)),joinpath(OUT,"irfs.png"))
    serialize(joinpath(OUT,"comparison.jls"),(responses=responses,names=names,ksbaseline=kb,nnbaseline=nb))
    println("COMPLETE. Baseline K drifts: KS=$(kb.K[end]-ss.K), NN=$(nb.K[end]-ss.K)")
end
main()
