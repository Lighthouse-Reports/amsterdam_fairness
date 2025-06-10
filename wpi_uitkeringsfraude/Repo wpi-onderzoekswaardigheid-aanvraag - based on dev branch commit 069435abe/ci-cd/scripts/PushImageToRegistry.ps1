$ModuleName = 'CCC.ContainerRegistry'
$ModulePath = Join-Path -Path "$env:BUILD_SOURCESDIRECTORY/BB-ContainerRegistry/" -ChildPath "./modules/$ModuleName"

Remove-Module $ModuleName -ErrorAction SilentlyContinue
Import-Module -Name $ModulePath

# Get the ARM connection service principle ID
$ctx = Get-AzContext
$prince = Get-AzADServicePrincipal -ApplicationId $ctx.Account.Id

# Assign role 'AcrPush', so that the pipeline is allowed to push the Docker image
$Parameters = @{
    ResourceGroupName     = "$env:DOCKERREGISTRYRESOURCEGROUPNAME"
    ContainerRegistryName = "$env:DOCKERREGISTRYNAME"
    ObjectId              = $prince.Id
    RoleDefinitionName    = 'AcrPush'
}
Set-ContainerRegistryAccessRole @Parameters

$ContainerRegistry = Get-AzContainerRegistry -ResourceGroupName "$env:DOCKERREGISTRYRESOURCEGROUPNAME" -Name "$env:DOCKERREGISTRYNAME"
$buildTag = "$env:DOCKERIMAGENAME" + ":" + "$env:DOCKERIMAGETAG"
$targetTag = "$($ContainerRegistry.LoginServer)/$env:DOCKERIMAGENAME" + ":" + "$env:DOCKERIMAGETAG"

# Rename the locally built image to a name that includes the registry
# We do not know the registry in the dockerBuild task
docker tag $buildTag $targetTag

# Pushing container image
$ContainerImageParameters = @{
    ResourceGroupName     = "$env:DOCKERREGISTRYRESOURCEGROUPNAME"
    ContainerRegistryName = "$env:DOCKERREGISTRYNAME"
    ImageName             = "$env:DOCKERIMAGENAME"
    ImageTag              = "$env:DOCKERIMAGETAG"
}
Push-ContainerImage @ContainerImageParameters

